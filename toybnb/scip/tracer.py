import numpy as np
import networkx as nx
import operator as op

from warnings import warn
from itertools import chain
from math import fsum, isclose
from heapq import heappop, heappush
from time import monotonic_ns

from numpy import ndarray
from pyscipopt import Model
from pyscipopt.scip import Node, Solution
from pyscipopt.scip import PY_SCIP_NODETYPE as SCIP_NODETYPE

from scipy.optimize import OptimizeResult

from ecole.core.scip import Model as ecole_Model

from .scip import SCIP_LPSOLSTAT_TO_NUMERIC
from ..tree import build_optresult, Status, DualBound

# SCIP_NODETYPE: form `type_tree.h`
# FOCUSNODE   =  0  # the focus node
# PROBINGNODE =  1  # temp. child node of the focus or refocused node used for probing
# SIBLING     =  2  # unsolved sibling of the focus node
# CHILD       =  3  # unsolved child of the focus node
# LEAF        =  4  # unsolved leaf of the tree, stored in the tree's queue
# DEADEND     =  5  # temporary type of focus node, if it was solved completely
# JUNCTION    =  6  # fork without LP solution
# PSEUDOFORK  =  7  # fork without LP solution and added rows and columns
# FORK        =  8  # fork with solved LP and added rows and columns
# SUBROOT     =  9  # fork with solved LP and arbitrarily changed rows and columns
# REFOCUSNODE = 10  # refocused for domain propagation (junction, fork, or subroot)


def get_sol_result(m: Model, sol: Solution, sign: float = 1.0) -> OptimizeResult:
    """Parse SCIP's solution and evaluate it."""

    # pyscipopt does not implement meaningful method for a solution, but
    #  thankfully, its repr is that of a dict, so it is kind of a proto-dict
    x = eval(repr(sol), {}, {})  # XXX dirty hack!

    # get the coefficients in linear objective and the offset
    # XXX it appears that getVars and sol's dict have variables is the same order
    obj = {v.name: v.getObj() for v in m.getVars(True)}
    val = fsum(obj[k] * v for k, v in x.items())
    c0 = m.getObjoffset(False)

    return build_optresult(x=x, fun=sign * (c0 + val), status=0, nit=-1)


class Tracer:
    sign: float
    is_worse: callable
    T: nx.DiGraph
    focus_: int
    nit_: int
    trace_: list
    duals_: list
    frontier_: set
    fathomed_: set

    def __init__(self, m: Model, *, ensure: str = None) -> None:
        """Initialize the tracer tree."""
        assert ensure is None or ensure in ("min", "max")

        # internally SCIP represents LPs as minimization problems
        sense = m.getObjectiveSense()[:3].lower()
        self.sign = -1.0 if sense == "max" else +1.0
        self.is_worse = op.lt if sense == "max" else op.gt

        # figure out the objective and how to compare the solutions
        # XXX actually we can use the `get_sol_result(m.getBestSol())`
        val = float("-inf" if sense == "max" else "+inf")
        inc = build_optresult(x={}, fun=val, nit=-1, status=0)

        self.duals_, self.trace_, self.focus_, self.nit_ = [], [], None, 0
        self.shadow_, self.frontier_, self.fathomed_ = None, set(), set()
        self.T = nx.DiGraph(root=None, incumbent=inc)

    def add_lineage(self, m: Model, n: Node) -> int:
        """Add node's representation to the tree and ensure its lineage exists."""
        assert isinstance(n, Node)
        if n.getType() not in (
            SCIP_NODETYPE.SIBLING,
            SCIP_NODETYPE.CHILD,
            SCIP_NODETYPE.LEAF,
            SCIP_NODETYPE.FOCUSNODE,
        ):
            raise NotImplementedError

        # add an un-visited node
        j = v = int(n.getNumber())
        if v not in self.T:
            # use the lower bound info (actually uninitialized until focused)
            fun = m.getObjoffset(False) + n.getLowerbound()

            self.T.add_node(
                v,
                type=n.getType(),  # SIBLING, LEAF, CHILD, FOCUSNODE
                lp=build_optresult(x={}, fun=self.sign * fun, status=0, nit=-1),
                best=None,
                status=Status.OPEN,
                n_visits=0,
                n_order=-1,  # monotonic visitation order (-1 -- never visited)
            )

        # if the node has been added earlier, then update its SCIP's type
        else:
            # SIBLING -> LEAF, SIBLING -> FOCUSNODE, LEAF -> FOCUSNODE
            self.T.add_node(v, type=n.getType())

        # continue unless we reach the root
        p = n.getParent()
        while p is not None:
            # guard against poorly understood node types
            if p.getType() not in (
                SCIP_NODETYPE.FOCUSNODE,
                SCIP_NODETYPE.FORK,
                SCIP_NODETYPE.SUBROOT,
                SCIP_NODETYPE.JUNCTION,
            ):
                raise NotImplementedError

            # add an un-visited node
            # XXX `add_edge` silently adds the endpoints, so we add them first
            u = int(p.getNumber())
            if u not in self.T:
                fun = m.getObjoffset(False) + p.getLowerbound()
                self.T.add_node(
                    u,
                    type=p.getType(),  # SIBLING, LEAF, CHILD, FOCUSNODE
                    lp=build_optresult(x={}, fun=self.sign * fun, status=0, nit=-1),
                    best=None,
                    status=Status.OPEN,
                    n_visits=0,
                    n_order=-1,
                )

            # if the node has been added earlier, then update its SCIP's type
            else:
                self.T.add_node(u, type=p.getType())

            # get the lp gain
            # XXX the lower bound is meaningless until the child is focused
            gain = max(self.sign * (n.getLowerbound() - p.getLowerbound()), 0)

            # establish or update the parent (u) child (v) link
            # XXX see [SCIP_BOUNDTYPE](/src/scip/type_lp.h#L44-50) 0-lo, 1-up
            # XXX the parent branching may not exist, when SCIP is shutting down
            dir, var, tau = None, None, None
            if n.getParentBranchings() is not None:
                (var,), (tau,), (uplo,) = n.getParentBranchings()
                dir = +1 if uplo > 0 else -1  # XXX same dir signs as in `toybnb.tree`
                var = var.getIndex()

            self.T.add_edge(u, v, key=dir, j=var, x=tau, g=gain)

            # ascend
            v, n, p = u, p, p.getParent()

        return j

    def enter(self, m: Model) -> int:
        """Begin processing the focus node at the current branching point."""

        # get the focus node
        # XXX We neither borrow nor own any `scip.Node` objects, but we are
        #  guaranteed that `n` references a valid focus node at least for the
        #  duration of the current branching call
        n: Node = m.getCurrentNode()
        assert isinstance(n, Node)

        # the focus node is not integer-feasible, since we got called,
        #  and is formally an open node, since it has just been the
        #  frontier node, and bnb does not revisit nodes
        assert n.getType() == SCIP_NODETYPE.FOCUSNODE

        # if we're visiting a former child/sibling/leaf make sure it is OPEN,
        #  and not shadow visited by SCIP, i.e. FATHOMED.
        if n.getParent() is not None and n.getNumber() in self.T:
            j = n.getNumber()
            if self.T.nodes[j]["status"] != Status.OPEN:
                raise NotImplementedError(
                    f"SCIP should not focus on non-open nodes. Got `{j}`."
                )

        # add the node to the tree and recover its LP solution
        # XXX SCIP guarantees that a node's number uniquely identifies a search
        #  node, even those whose memory SCIP reclaimed
        j = self.focus_ = self.add_lineage(m, n)  # OPEN

        # use monotonic clock, which cannot go backward, for recording the
        #  focus/visitation order
        self.T.nodes[j]["n_order"] = monotonic_ns()

        # the root may get visited twice
        if n.getParent() is not None and self.T.nodes[j]["n_visits"] > 0:
            raise NotImplementedError(
                f"SCIP should not revisit nodes, other than the root. Got `{j}`."
            )

        # Extract the local LP solution at the focus node
        # XXX the LP seems to be always `\min`
        trans = True
        x = {v.name: v.getLPSol() for v in m.getVars(trans)}
        val = fsum(v.getLPSol() * v.getObj() for v in m.getVars(trans))
        assert isclose(val, n.getLowerbound())  # XXX lb is w.r.t. `\min`

        # we hope, the offset has the correct sign for the LPs inside nodes
        c0 = m.getObjoffset(not trans)  # XXX to be used for comparing with incumbent

        # the lp solution at the focus node cannot be integer feasible, since
        #  otherwise we would not be called in the first place
        lp = self.T.nodes[j]["lp"] = build_optresult(
            x=x,
            fun=self.sign * (c0 + val),
            status=SCIP_LPSOLSTAT_TO_NUMERIC[m.getLPSolstat()],
            nit=m.getNLPIterations() - self.nit_,
        )

        # XXX technically we don't need to store the root node, since there can
        # only be one
        if self.T.graph["root"] is None:
            self.T.graph["root"] = j

        # maintain our own pruning pq
        # XXX the lp value of a node is not available until it is in focus, so
        #  we do not do this, when enumerating the open frontier
        heappush(self.duals_, DualBound(-lp.fun, j))

        # then current focus node may not have been designated by us as an open
        #  node, since it is the immediate child of the last focus node, and we
        #  can only see the open node set as it is after focusing, but before
        #  branching
        if j in self.frontier_:
            self.frontier_.remove(j)

        else:
            # j's parent is the focus node that we visited immediately before
            pass

        self.nit_ = m.getNLPIterations()

        self.T.nodes[j]["n_visits"] += 1
        return j

    def leave(self, m: Model) -> None:
        """Conclude the data collection for the last focus node from SCIP's state
        that was revealed upon calling branching at a new focus node.
        """
        assert self.focus_ is not None

        # close the focus node from our previous visit, since SCIPs bnb never
        #  revisits
        self.T.nodes[self.focus_]["status"] = Status.CLOSED

    def prune(self) -> None:
        """SCIP shadow-fathoms the nodes for us. We attempt to recover, which
        nodes were fathomed by pruning.
        """

        nodes = self.T.nodes
        while self.duals_ and (self.T.graph["incumbent"].fun < -self.duals_[0].val):
            node = heappop(self.duals_).node
            # do not fathom nodes, re-fathomed by SCIP
            if nodes[node]["status"] == Status.FATHOMED:
                continue

            assert nodes[node]["status"] in (Status.OPEN, Status.CLOSED)
            nodes[node]["status"] = Status.PRUNED

    def add_frontier(self, m: Model) -> set:
        """Update the set of tracked open nodes and figure out shadow-visited ones."""

        # ensure all currently open nodes from SCIP are reflected in the tree
        # XXX [xternal.c](scip-8.0.1/doc/xternal.c#L3668-3686) implies that the
        #  other getBest* methods pick nodes from the open (unprocessed) frontier
        new_frontier = set()
        for n in chain(*m.getOpenNodes()):
            new_frontier.add(self.add_lineage(m, n))
            # XXX We do not add to the dual pq here, becasue the leaf, child
            #  and sibling nodes appear to have uninitialized default lp values

        # if the current set of open nodes is not a subset of the open nodes
        #  upon processing the previous focus node, then SCIP in its solver
        #  loop visited more than one node focus node, before asking for a
        #  branchrule's decision.
        shadow = self.frontier_ - new_frontier
        for j in shadow:
            # the shadow nodes are all nodes processed by SCIP
            #  in between consecutive calls to var branching. Each
            #  could've been PRUNED, or marked as FEASIBLE/INFEASIBLE.
            #  One way or the other they're FATHOMED.
            self.T.nodes[j]["status"] = Status.FATHOMED

        self.frontier_ = new_frontier
        return shadow

    def update(self, m: Model, fin: bool = False) -> None:
        """Update the tracer tree."""

        # finish processing the last focus
        if self.focus_ is not None:
            self.leave(m)

        # start processing the current focus node, unless the search has finished
        # XXX we check `fin` flag, since occasionally the current node may not
        #  be none when BnB is finished (a memleak?)
        if not fin and m.getCurrentNode() is not None:
            j = self.enter(m)

            # record the path through the tree
            self.trace_.append(
                (
                    j,
                    m.getPrimalbound(),  # self.T.graph["incumbent"].fun,
                    self.T.nodes[j]["lp"].fun,  # XXX not m.getDualbound()
                )
            )

        else:
            # clear the focus node when SCIP terminates the bnb search
            self.focus_ = None
            if m.getCurrentNode() is not None:
                warn("Ecole's `fin=True` with SCIP's non-None focus.", RuntimeWarning)

        # track the best sol maintained by SCIP
        # XXX While processing branching takes place at [SCIPbranchExecLP](solve.c#4420)
        # Therefore, the current best, if it has been updated, should be attributed
        # to a previous focus node or to a node kindly fathomed for us by SCIP,
        #  however. not all solutions get into SCIP's storage
        # XXX [addCurrentSolution](solve.c#5039) the integer-feasible solution
        #  is added after the focus node is processed [solveNode](solve.c#4982).
        #  [primalAddSol](primal.c#1064)
        sols = m.getSols()
        if sols:
            lp = get_sol_result(m, sols[0], self.sign)
            if self.is_worse(self.T.graph["incumbent"].fun, lp.fun):
                self.T.graph["incumbent"] = lp

        self.prune()

        # attributing the best bound chance to the last focus is not ideal, since
        #  branchrule is not called when SCIP's LP solver detected integer feasibility
        #  or overall infeasibility. A good thing is that such nodes were open in
        #  the past, so we must recompute their fate.
        self.shadow_ = shadow = self.add_frontier(m)
        self.fathomed_.update(shadow)

        if self.focus_ is None:
            more_than_two = [n for n in self.T if len(self.T[n]) > 2]
            if any(more_than_two):
                raise NotImplementedError(
                    f"Some nodes have more than two children: `{more_than_two}`."
                )

            # SCIP's `.getNTotalNodes` reports the total number of nodes generated
            #  by ITS search. Some of these nodes are shadow-visited, i.e. we
            #  were never allowed to act in, since SCIP processed them internally,
            #  but which we nevertheless saw through our frontier recovery logic.
            #  Other nodes shadow-visited are those, which SCIP processsed IMMEDIATELY
            #  UPON CREATION, i.e. nodes with integer-feasible or infeasible
            #  sub-problems created due to var-splitting but immediately fathomed.
            # In our projection of SCIP's tree the latter nodes are the visited
            #  nodes, which have less than two children.
            # XXX `fathomed_` are the nodes from the former category.
            # XXX we test for upper bound, since SCIP may not consider certain
            #  sub-problems as nodes.
            visited = [n for n, v in self.T.nodes.items() if v["n_visits"] > 0]
            if len(self.fathomed_) + len(visited) != len(self.T):
                # XXX it is sometimes possible for scip to have fin=True and
                #  a focus node
                raise RuntimeError("Corrupted tree")

            # we can at most overcount the number of scip's nodes
            n_expected = len(self.T) + sum(2 - len(self.T[n]) for n in visited)
            if self.T and n_expected < m.getNTotalNodes():
                warn(
                    f"Node accounting error: {n_expected} < {m.getNTotalNodes()}",
                    RuntimeWarning,
                )


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
        self.tracer.update(model.as_pyscipopt(), fin)

        if not fin:
            return None

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
