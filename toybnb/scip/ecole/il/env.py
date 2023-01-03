import numpy as np
from numpy import _NoValue, ndarray
from numpy.random import SeedSequence

import torch
from torch import Tensor, as_tensor

import pyscipopt  # noqa: F401

import ecole as ec  # noqa: F401
from ecole import RandomGenerator, reward
from ecole.environment import Branching
from ecole.observation import NodeBipartite, NodeBipartiteObs

from typing import NamedTuple


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


class Observation(NamedTuple):
    obs: NodeBipartiteObs
    actset: ndarray


class BatchObservation(NamedTuple):
    # variable, constraint, and cons-to-vars link features
    vars: Tensor
    cons: Tensor

    ctov_v: Tensor
    ctov_ij: Tensor

    actset: Tensor

    # the first index in the collated batch of the span of data
    #  originating from each uncolated batch element
    ptr_vars: Tensor
    ptr_cons: Tensor
    ptr_ctov: Tensor
    ptr_actset: Tensor

    # batch affinity
    inx_vars: Tensor
    inx_cons: Tensor
    inx_ctov: Tensor
    inx_actset: Tensor


def collate(batch: tuple[Observation], device: torch.device = None) -> BatchObservation:
    """Collate `NodeBipartiteObs` into torch tensors"""
    obs, actset = zip(*batch)

    # prepare vars
    vars = [x.variable_features for x in obs]
    n_vars = sum(map(len, vars))
    x_vars = torch.empty(
        (n_vars, *vars[0].shape[1:]),
        dtype=torch.float32,
        device=device,
    )

    ptr_vars = x_vars.new_zeros(1 + len(vars), dtype=torch.long)
    inx_vars = x_vars.new_empty(n_vars, dtype=torch.long)

    # prepare cons
    cons = [x.row_features for x in obs]
    n_cons = sum(map(len, cons))
    x_cons = x_vars.new_empty((n_cons,) + cons[0].shape[1:])

    ptr_cons = x_cons.new_zeros(1 + len(cons), dtype=torch.long)
    inx_cons = x_cons.new_empty(n_cons, dtype=torch.long)

    # prepare edges (coo ijv, cons-to-vars)
    ctov = [x.edge_features for x in obs]
    n_ctov = sum(len(e.values) for e in ctov)
    x_ctov_v = x_vars.new_empty(n_ctov)
    x_ctov_ij = x_vars.new_empty((2, n_ctov), dtype=torch.long)

    ptr_ctov = x_ctov_v.new_zeros(1 + len(ctov), dtype=torch.long)
    inx_ctov = x_ctov_v.new_empty(n_ctov, dtype=torch.long)

    # prepare the collated action set
    n_actset = sum(map(len, actset))
    x_actset = x_vars.new_empty(n_actset, dtype=torch.long)

    ptr_actset = x_actset.new_zeros(1 + len(actset), dtype=torch.long)
    inx_actset = x_actset.new_empty(n_actset, dtype=torch.long)

    # copy numpy data into the allocated tensors
    v1 = c1 = e1 = j1 = 0
    for b, (x, act_set) in enumerate(batch):
        v0, v1 = v1, v1 + len(x.variable_features)
        c0, c1 = c1, c1 + len(x.row_features)
        e0, e1 = e1, e1 + len(x.edge_features.values)
        j0, j1 = j1, j1 + len(act_set)

        # the vars, cons, and cons-to-vars (edges)
        x_vars[v0:v1].copy_(as_tensor(x.variable_features))
        x_cons[c0:c1].copy_(as_tensor(x.row_features))
        x_ctov_v[e0:e1].copy_(as_tensor(x.edge_features.values))
        x_ctov_ij[:, e0:e1].copy_(as_tensor(x.edge_features.indices.astype(int)))
        x_actset[j0:j1].copy_(as_tensor(act_set.astype(int)))

        # fixup the ij-link indices and action set
        x_ctov_ij[0, e0:e1] += c0
        x_ctov_ij[1, e0:e1] += v0
        x_actset[j0:j1] += v0

        # the batch assignment
        inx_vars[v0:v1] = b
        inx_cons[c0:c1] = b
        inx_ctov[e0:e1] = b
        inx_actset[j0:j1] = b

        # record the batch index pointer
        ptr_vars[1 + b] = v1
        ptr_cons[1 + b] = c1
        ptr_ctov[1 + b] = e1
        ptr_actset[1 + b] = j1

    return BatchObservation(
        x_vars,
        x_cons,
        x_ctov_v,
        x_ctov_ij,
        x_actset,
        ptr_vars,
        ptr_cons,
        ptr_ctov,
        ptr_actset,
        inx_vars,
        inx_cons,
        inx_ctov,
        inx_actset,
    )
