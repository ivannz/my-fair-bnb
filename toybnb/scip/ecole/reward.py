import numpy as np
import networkx as nx

from warnings import warn

from numpy import ndarray
from ecole.core.scip import Model as ecole_Model

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

    def before_reset(self, model: ecole_Model) -> None:
        self.tracer = Tracer(model.as_pyscipopt())

    def extract(self, model: ecole_Model, fin: bool) -> ndarray:
        m = model.as_pyscipopt()
        self.tracer.update(m, fin)

        if not fin:
            return None

        if m.getCurrentNode() is not None:
            warn("Ecole's `fin=True` with SCIP's non-None focus.", RuntimeWarning)

        T = self.tracer.T

        # the instance was pre-solved if the traced could not find the root
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
