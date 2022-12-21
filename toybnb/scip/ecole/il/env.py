import ecole as ec  # noqa: F401
import numpy as np
import pyscipopt  # noqa: F401

from ecole import RandomGenerator, reward
from ecole.environment import Branching
from ecole.observation import NodeBipartite

from numpy import _NoValue
from numpy.random import SeedSequence


def default_scip_params(*, time: int = 3600) -> dict:
    return {
        # although we use ecole's `.disable_presolve()`, we still keep these params
        "separating/maxrounds": 0,  # separate (cut) only at root node
        "presolving/maxrestarts": 0,  # disable solver restarts
        # determines scip's inner clock and affects the time limit
        "timing/clocktype": 1,  # 1: CPU user seconds, 2: wall clock time
        "limits/time": time,  # solver time limit
    }


class BranchingWithoutPresolve:
    def reset(self, instance, *dynamics_args, **dynamics_kwargs):
        # disable presolve through ecole
        # XXX [.disable_presolve](./libecole/src/scip/model.cpp#L195)
        #  calls [SCIPsetPresolving](./src/scip/scip_params.c#L913-937)
        instance = instance.copy_orig()  # XXX .reset also makes a copy
        instance.disable_presolve()

        return super().reset(instance, *dynamics_args, **dynamics_kwargs)


def make_env(
    entropy: int = None, presolve: bool = True, scip_params: dict = _NoValue
) -> Branching:
    # fork the seed sequence from the given entropy
    ss = entropy if isinstance(entropy, SeedSequence) else SeedSequence(entropy)

    # allow for `true None` scip-params
    if scip_params is _NoValue:
        scip_params = default_scip_params()

    # choose the env
    cls = Branching if presolve else BranchingWithoutPresolve

    # the branching env
    env = cls(
        # We use bipartite graph repr of the node's LP
        observation_function=NodeBipartite(),
        # No reward function at, since we imitate an expert
        # reward_function=reward.PrimalDualIntegral(),
        reward_function=reward.Constant(float("nan")),
        # we track the aggregate tree stats
        # XXX not sure if `nnodes` is a `clean` metric
        information_function={
            "n_nodes": reward.NNodes().cumsum(),
            "n_lpiter": reward.LpIterations().cumsum(),
            "f_soltime": reward.SolvingTime().cumsum(),
            # 'primal_integral': reward.PrimalIntegral().cumsum(),
            # 'dual_integral': reward.DualIntegral().cumsum(),
            # 'primal_dual_integral': reward.PrimalDualIntegral(),
        },
        scip_params=scip_params,
    )

    # `RandomGenerator.max_seed` reports 2^{32}-1
    (seed,) = ss.generate_state(1, dtype=np.uint32)
    env.seed(int(seed))

    return env


def ecole_seed(ss: SeedSequence) -> RandomGenerator:
    """Seeded Ecole's PRNG object from a proper entropy soruce"""
    # `RandomGenerator.max_seed` reports 2^{32}-1
    (seed,) = ss.generate_state(1, dtype=np.uint32)
    return RandomGenerator(seed)
