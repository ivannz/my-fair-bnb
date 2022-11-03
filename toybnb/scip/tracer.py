import networkx as nx
from pyscipopt import scip

from operator import lt, gt
from itertools import chain

from ecole.environment import Branching
from functools import wraps


def add_node(G: nx.DiGraph, n: scip.Node) -> int:
    node: int = int(n.getNumber())
    # add the node to the tree
    if node not in G:
        G.add_node(node, lb=n.getLowerbound(), n_visits=0)

    # find out the parent's branchings
    p: scip.Node = n.getParent()
    if p is not None:
        # the parent must've been visited eariler
        parent: int = int(p.getNumber())
        assert parent in G

        (var,), (tau,), (d,) = n.getParentBranchings()
        G.add_edge(parent, node, var=(var.getIndex(), d, tau))

    return node


class SCIPTreeTracer:
    """Trace visited nodes and the frontier."""

    def __init__(self, m: scip.Model) -> None:
        self.tree_, self.focus_ = nx.DiGraph(), None

        # best incumbent tracking
        sense = m.getObjectiveSense()[:3].lower()
        self.best_ = float("+inf" if sense == "min" else "-inf")
        self.op_ = lt if sense == "min" else gt
        self.trace_ = []

    def update(self, m: scip.Model) -> None:
        assert isinstance(m, scip.Model)

        # XXX do not store pointers to nodes!
        primalbound = m.getPrimalbound()

        # detect incumbent improvement: it has improved since processing
        #  the last node, hence the last node should take credit for the
        #  new best solution
        if self.op_(primalbound, self.best_):
            self.best_ = primalbound
            if self.focus_ is not None:
                self.tree_.nodes[self.focus_]["best"] = self.best_

        # visit the focus node unless bnb is done
        n: scip.Node = m.getCurrentNode()
        if n is None:
            return

        # add the node to the tree
        self.focus_ = add_node(self.tree_, n)

        # prefetch the frontier
        for o in chain(*m.getOpenNodes()):
            add_node(self.tree_, o)

        self.tree_.nodes[self.focus_]["n_visits"] += 1
        self.trace_.append((self.focus_, primalbound, n.getLowerbound()))


class TracedBranching(Branching):
    tracer: SCIPTreeTracer

    @wraps(Branching.reset)
    def reset(self, instance, *dynamics_args, **dynamics_kwargs):
        try:
            self.tracer_ = SCIPTreeTracer(instance.as_pyscipopt())
            return super().reset(instance, *dynamics_args, **dynamics_kwargs)
        finally:
            self.tracer_.update(self.model.as_pyscipopt())

    @wraps(Branching.step)
    def step(self, action, *dynamics_args, **dynamics_kwargs):
        try:
            return super().step(action, *dynamics_args, **dynamics_kwargs)
        finally:
            self.tracer_.update(self.model.as_pyscipopt())
