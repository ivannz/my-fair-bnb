from functools import wraps
from typing import Callable, Iterable

import numpy as np
import torch

from ecole.core.scip import Stage
from ecole.environment import Branching
from ecole.observation import Pseudocosts, StrongBranchingScores
from numpy.random import default_rng
from torch import Tensor

from queue import SimpleQueue, Empty
from threading import Thread, Event

from .data import BatchObservation, Observation, collate

BranchRule = Callable[[Branching, dict], int]
BranchRuleCallable = Callable[[Observation], int]


def strongbranch(pseudocost: bool = False) -> BranchRule:
    """Return the strong branching policy"""

    def _spawn(env: Branching, config: dict = None) -> BranchRuleCallable:
        """Get the strong branching policy for env"""
        if not pseudocost:
            scorer = StrongBranchingScores(pseudo_candidates=False)

        else:
            scorer = Pseudocosts()

        def _branchrule(obs: Observation) -> int:
            """Decide which variable to branch on using the SB heuristic"""
            if env.model.stage != Stage.Solving:
                return None

            scores = scorer.extract(env.model, False)
            return obs.actset[scores[obs.actset].argmax()]  # SCIPvarGetProbindex

        return _branchrule

    return _spawn


def randombranch(seed: int = None) -> BranchRule:
    """Return the policy of random branching"""

    def _spawn(env: Branching, config: dict = None) -> BranchRuleCallable:
        """Get the random branching rule for env"""
        rng = default_rng(seed)

        def _branchrule(obs: Observation) -> int:
            """Randomly pick which variable to branch on"""
            if env.model.stage != Stage.Solving:
                return None

            return int(rng.choice(obs.actset))

        return _branchrule

    return _spawn


class BaseNeuralBranchRuleMixin:
    """The base class for neural variable branching models"""

    def compute(
        self, input: BatchObservation, target: Tensor = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError

    def predict(self, input: BatchObservation) -> Tensor:
        raise NotImplementedError


def batched_ml_branchrule(
    module: BaseNeuralBranchRuleMixin, device: torch.device = None, config: dict = None
) -> BranchRuleCallable:
    """Create a batched brancher from a special neural model"""
    assert isinstance(module, BaseNeuralBranchRuleMixin)

    # make a shallow copy of the config dict
    config = dict({} if not isinstance(config, dict) else config)

    @wraps(module.predict)
    def _branchrule(batch: Iterable[Observation]) -> Iterable[int]:
        """Choose a batch of branching variables using the neural model"""
        module.eval()
        out = module.predict(collate(batch, device), **config)
        return np.asarray(out.cpu(), dtype=int).tolist()

    return torch.inference_mode(mode=True)(_branchrule)


def ml_branchrule(
    module: BaseNeuralBranchRuleMixin, device: torch.device = None
) -> BranchRule:
    """Create a brancher from a neural branching model"""
    assert isinstance(module, BaseNeuralBranchRuleMixin)

    def _spawn(env: Branching, config: dict = None) -> BranchRuleCallable:
        """Get a neural branching policy for env"""
        do_batch = batched_ml_branchrule(module, device, config)

        def _branchrule(obs: Observation) -> int:
            """Select a branching variable using the neural network"""
            if env.model.stage != Stage.Solving:
                return None

            # apply the model to a single-item batch
            return int(do_batch([obs])[0])

        return _branchrule

    return _spawn


class GenericBatchingServer(Thread):
    """Collect batches of requests and process them with `target`"""

    exception_: BaseException = None

    def __init__(
        self,
        target: Callable,
        *,
        name: str = None,
        daemon: bool = None,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        self.requests, self.is_finished = SimpleQueue(), Event()
        self.target, self.conections = target, []

    def run(self) -> None:
        try:
            # collect the first element via a short-lived blocking call
            for request in iter(self.requests.get, None):
                batch = [request]
                try:
                    # fetch the remaining __immediately__ available items
                    batch.extend(iter(self.requests.get_nowait, None))

                except Empty:
                    pass

                # process and send each result back to its origin, auto-shutdown
                #  in the case of emergency (`.target` throws)
                puts, inputs = zip(*batch)
                for put, out in zip(puts, self.target(inputs)):
                    put(out)

        except BaseException as e:
            self.exception_ = e

        finally:
            # make sure to inform all waiting `co_yield`-s about termination
            self.is_finished.set()

    def connect(self) -> Callable:
        """Establish a communications closure"""
        com = SimpleQueue()
        self.conections.append(com)

        def co_yield(input: ...) -> ...:
            # send the request, unless the server has been terminated
            if self.is_finished.is_set():
                raise RuntimeError

            self.requests.put((com.put, input))

            # wait on our exclusive queue (blocking)
            # XXX return the result unliess we get None
            for result in iter(com.get, None):
                return result

            raise self.exception_ or RuntimeError

        return co_yield

    def close(self):
        for com in self.conections:
            com.put(None)

        self.join()


class BranchingServer(GenericBatchingServer):
    """Branching variable Server"""

    def __init__(
        self, module: BaseNeuralBranchRuleMixin, device: torch.device = None
    ) -> None:
        super().__init__(batched_ml_branchrule(module, device))

    def connect(self, env: Branching, config: dict = None) -> BranchRuleCallable:
        """Spawn a new branchrule"""
        co_yield = super().connect()

        def _branchrule(obs: Observation) -> int:
            if env.model.stage != Stage.Solving:
                return None

            return int(co_yield(obs))

        return _branchrule
