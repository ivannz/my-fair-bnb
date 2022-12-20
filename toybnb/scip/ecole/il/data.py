from typing import NamedTuple

import torch
from ecole.observation import NodeBipartiteObs
from numpy import ndarray
from torch import Tensor, as_tensor


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
