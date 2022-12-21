from functools import wraps
from typing import Callable, Iterable

import numpy as np
import torch

from ecole.core.scip import Stage
from ecole.environment import Branching
from ecole.observation import Pseudocosts, StrongBranchingScores
from numpy.random import default_rng
from torch import Tensor

from .data import BatchObservation, Observation, collate

BranchRule = Callable[[Branching], int]
BranchRuleCallable = Callable[[Observation], int]


def strongbranch(pseudocost: bool = False) -> BranchRule:
    if not pseudocost:
        scorer = StrongBranchingScores(pseudo_candidates=False)

    else:
        scorer = Pseudocosts()

    def _spawn(env: Branching) -> BranchRuleCallable:
        def _branchrule(obs: Observation, **ignored) -> int:
            if env.model.stage != Stage.Solving:
                return None

            scores = scorer.extract(env.model, False)
            return obs.actset[scores[obs.actset].argmax()]  # SCIPvarGetProbindex

        return _branchrule

    return _spawn


def randombranch(seed: int = None) -> BranchRule:
    rng = default_rng(seed)

    def _spawn(env: Branching) -> BranchRuleCallable:
        def _branchrule(obs: Observation, **ignored) -> int:
            if env.model.stage == Stage.Solving:
                return int(rng.choice(obs.actset))
            return None

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


def batched_ml_branchrule(module: BaseNeuralBranchRuleMixin) -> BranchRuleCallable:
    @wraps(module.predict)
    def _branchrule(batch: Iterable[Observation], **kwargs) -> Iterable[int]:
        module.eval()
        out = module.predict(collate(batch), **kwargs).cpu()
        return np.asarray(out, dtype=int).tolist()

    return torch.inference_mode(mode=True)(_branchrule)


def ml_branchrule(module: BaseNeuralBranchRuleMixin) -> BranchRule:
    do_batch = batched_ml_branchrule(module)

    def _spawn(env: Branching) -> BranchRuleCallable:
        def _branchrule(obs: Observation, **ignored) -> int:
            if env.model.stage != Stage.Solving:
                return None

            # apply the model to a single-item batch
            return int(do_batch([obs])[0])

        return _branchrule

    return _spawn
