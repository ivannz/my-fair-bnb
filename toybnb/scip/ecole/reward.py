import numpy as np
import networkx as nx

from numpy import ndarray
from ecole.core.scip import Model

from ..tracer import Tracer


def subtree_size(T: nx.DiGraph, n: int) -> int:
    """Recursively compute the sizes of all sub-trees."""
    size = 1
    for c in T[n]:
        assert n != c
        size += subtree_size(T, c)

    T.nodes[n]["n_size"] = size
    return size


class NegLogTreeSize:
    """Reward/Information function with bnb tree tracing for Ecole's branching env."""

    tracer: Tracer

    def __init__(self) -> None:
        self.tracer = None

    def before_reset(self, model: Model) -> None:
        self.tracer = Tracer(model.as_pyscipopt())

    def extract(self, model: Model, fin: bool) -> ndarray:
        m = model.as_pyscipopt()
        self.tracer.update(m)

        # ecole will indicate `fin=True` while `m.getCurrentNode() is not None`
        #  in the case when a limit is reached (gap, nodes, iter, etc)
        if not fin:
            return None

        T = self.tracer.T

        # the instance was pre-solved if the tracer could not find the root
        if T.graph["root"] is None:
            return np.zeros(0, dtype=np.float32)

        # ensure correct tree size (including shadow-visited nodes)
        subtree_size(T, T.graph["root"])
        n_size = nx.get_node_attributes(T, "n_size")

        # the list of visited nodes ordered according to visitation sequence
        n_visits = nx.get_node_attributes(T, "n_visits")
        visited = [n for n in T if n_visits[n] > 0]

        n_order = nx.get_node_attributes(T, "n_order")
        assert all(n_order[n] >= 0 for n in visited)
        visited = sorted(visited, key=n_order.get)

        # fetch all visited nodes, which bore no children
        # XXX could've checked for `n_size[n] == 1` just as well
        # visited_leaves = [n for n in visited if not T[n]]
        # XXX these are the nodes makred as fathomed by Parsonson

        return -np.log(np.array([n_size[n] for n in visited], dtype=np.float32))


class LPGains:
    """Reward for Ecole's branching env based on dual bound gains."""

    tracer: Tracer

    def __init__(self, gamma: float = 0.25, pseudocost: bool = False) -> None:
        self.gamma, self.pseudocost = gamma, pseudocost
        self.tracer = None

    def before_reset(self, model: Model) -> None:
        self.tracer = Tracer(model.as_pyscipopt())

    def extract(self, model: Model, fin: bool) -> ndarray:
        m = model.as_pyscipopt()
        self.tracer.update(m)

        if not fin:
            return None

        # the instance was pre-solved if the tracer could not find the root
        T = self.tracer.T
        if T.graph["root"] is None:
            return np.zeros(0, dtype=np.float32)

        # the list of visited nodes ordered according to visitation sequence
        n_visits = nx.get_node_attributes(T, "n_visits")
        visited = [n for n in T if n_visits[n] > 0]

        n_order = nx.get_node_attributes(T, "n_order")
        visited = sorted(visited, key=n_order.get)
        assert all(n_order[n] >= 0 for n in visited)

        # Get the ell-1 norm of the coefficients of the linear objective
        # XXX we hope that SCIP does not re-scale the sub-problems
        obj = {v.name: v.getObj() for v in m.getVars(transformed=True)}
        scale = max(map(abs, obj.values()))  # sum

        # compute the normalized lp gains on edges between visited nodes
        lps, scores = nx.get_node_attributes(T, "lp"), []
        for u in visited:
            score, k = 1.0, 0
            for v, dt in T[u].items():
                if n_visits[v] < 1 or dt["g"] <= 0.0:
                    continue

                if self.pseudocost:
                    s = dt["g"] / dt["f"]
                    # XXX what is a good natural scale for the pseudocosts?
                    # s = min(max(dt["g"] / dt["f"], 0.0), 1.0)

                else:
                    # \Delta_\pm = c^\top (x_\pm - x) \leq \|c\|_p \|x_\pm - x\|_q
                    # get the ell-infty norm between nested solutions
                    vx, ux = lps[v].x, lps[u].x
                    s = dt["g"] / (scale * sum(abs(vx[k] - ux[k]) for k in obj))

                score *= s
                k += 1

            # compute the geometric mean (`k` is at most 2)
            scores.append(score ** (1 / max(k, 1.0)))  # XXX in [0, 1]
            # XXX after tree tracing we may end up with visited leaves, or visited
            #  nodes with just one child, because SCIP can immediately fathom
            #  children after branching, without ever informing us, due to primal
            #  bound cutoff, infeasibility, or interger-feasibility
            #  - such nodes are good, because they lead to immediate fathoming,
            #    hence they get a +1 reward

        return np.array(scores, dtype=np.float32) ** self.gamma
