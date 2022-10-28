import numpy as np
import networkx as nx

from scipy.optimize import linprog, OptimizeResult
from numpy.random import default_rng
from heapq import heappush, heappop  # noqa: F401
from math import floor, ceil

from numpy import ndarray
from collections import namedtuple
from enum import Enum

from .milp import MILP, is_feasible_int


def lpsolve(p: MILP) -> tuple[ndarray, float]:
    """Drop the integrality constraints and solve the relaxed LP."""
    # fire the scipy's lp solver for \min_x \{ c^\top x : A x \leq b, x \in [l, u]\}
    # XXX "interior-point", "simplex", and "revised simplex" are deprecated
    # XXX linprog accepts bounds in N x 2 ndarray form as well, with unbounded
    #  variables having `\pm\infty` bounds.
    #  [_clean_inputs](scipy/optimize/_linprog_util.py#L390-452)
    lp = linprog(p.c, p.A_ub, p.b_ub, p.A_eq, p.b_eq, p.bounds, method="highs")
    if lp.success:
        return lp

    # the data in OptimizeResult in case of an error status varies depending
    #  on the method, thus we construct our own result
    # XXX `.x` is x^* the solution and `.fun` is c^\top x^*
    return OptimizeResult(
        x=None,
        # XXX `lp.status` codes ( `lp.success = (lp.status == 0)`)
        #  0 sucess, 1 iter limit, 2 infeasible, 3 unbounded, 4 numerical
        fun=+np.inf if lp.status == 2 else -np.inf,
        success=False,
        status=lp.status,
        message=lp.message,
        nit=lp.nit,
    )


def split(p: MILP, by: int, threshold: float) -> (MILP, MILP):
    r"""Split the problem into halves by the specified integer variable.

    Let $
        S = \{
            Ax \leq b
            \,, x \in [l, u]
            \,, x \in \mathbb{Z}^m \times \mathbb{R}^{n-m}
        \}
    $ be the feasibility set of a MILP, and $
        S' = \{
            Ax \leq b
            \,, x \in [l, u]
        \}
    $ be its relaxation (continuous).

    Let $
        x \in \arg\min\{c^\top x \colon x \in S'\}
    $ be an lp solution and assume that the set $
        F_x \subseteq \{1..m\}
    $ of variables that are supposed to take integer values, but ended up being
    fractional, is non-empty. Then each $j \in F$ yields the following axis-aligned
    binary split of the original region $S$:
    $$
    L_j = \bigl[l_j, \lfloor x_j \rfloor \bigr] \times \mathbb{R}^{n-1}
        \,, $$
    and
    $$
    R_j = \bigl[\lceil x_j \rceil, u_j \bigr] \times \mathbb{R}^{n-1}
        \,. $$
    Careful observation reveals that
    $$
    S \setminus \bigl(S \cap (L_j \uplus R_j) \bigr)
        = S
        \cap \Bigl(
            \bigl(\lfloor x_j\rfloor, \lceil x_j\rceil \bigr) \times \mathbb{R}^{n-1}
        \Bigr)
        = \emptyset
        \,, $$
    since the $j$-th variable must take integer values.
    """
    lo, hi = p.bounds[by]

    # `lo` sub-problem [l_j, \lfloor \tau \rfloor]
    b_lo = p.bounds.copy()
    b_lo[by, 1] = floor(threshold)

    # `hi` sub-problem [\lceil \tau \rceil, u_j]
    b_hi = p.bounds.copy()
    b_hi[by, 0] = ceil(threshold)

    # make a shallow copy
    return p._replace(bounds=b_lo), p._replace(bounds=b_hi)


class Status(Enum):
    """Node state status codes.

    Meaning
    -------
    OPEN:
        A sub-problem has been created and its dual value computed. It's current
        lp value (dual lower bound) is better, than the current global incumbent
        solution. The node still has some branching options left to explore.
    CLOSED:
        All branching options offered by this node have been exhausted, but its
        sub-tree has not necessarily been fathomed.
    PRUNED:
        The node had to be eliminated due to certifiable sub-optimality. During
        the search a non overlapping sub-problem produced an integer-feasible
        solution with a LOWER objective value, than this node's lp relaxation,
        which, by design, bounds any integer-feasible solution in its region.
    FEASIBLE:
        The lp relaxation of the node's sub-problem is integer-feasible and
        optimal, hence there is no reason to dive into its sub-tree. The branch
        is fathomed.
        Note that a feasible node can reside in a pruned branch, in which case
        it is also considered pruned.
    INFEASIBLE:
        The relaxed lp has a degenerate feasibility region, meaning that node's
        the MILP sub-problem is infeasible. Again, the sub-tree is fathomed.

    Closed nodes
    ------------
    If the node's lp solution is integer infeasible, then the subset of the
    node's feasibility region NOT COVERED by any axis-aligned binary split on
    fractional variables is excluded from all sub-problems in the sub-tree. This
    region is an open rectangle, possibly unbounded, with at least one side
    being an open interval between CONSECUTIVE integers. Thus, this region can
    be safely exculed from consideration, since is cannot contain any integer
    feasible solutions to the node's problem.

    A sub-problem in a partition of a binary split either may be immediately
    infeasible (empty feasibility set), or may have produced an integer feasible
    candidate after diving after exhausting all branching options. Although the
    current node's lp lower bound might still have a large gap with respect to
    the primal bound, we can safely mark this node as fathomed.

    Solution for the node's integer sub-problem
    -------------------------------------------
    If children of this node produced integer feasible solutions, then
    we can select the best among them and assign it to this node.
    """

    OPEN = 0
    CLOSED = 5

    PRUNED = 1
    FEASIBLE = 2
    INFEASIBLE = 3


DualBound = namedtuple("DualBound", "val,node")


def init(p: MILP, *, seed: int = None) -> nx.DiGraph:
    """Initialize the branch and bound search tree."""
    inc = OptimizeResult(x=None, fun=np.inf, success=False, status=1, message="")

    # init the tree graph and add the root node with the original problem
    return nx.DiGraph(
        None,
        # the original primal linear problem
        p=p,
        # the best integer feasible solution found so far
        incumbent=inc,
        # the queue for sub-problem prioritization
        queue=[],  # deque([]),
        # the max heap of dual bounds for pruning
        # XXX we store bounds in `DualBound` nt-s with value of the opposite
        #  sign, since heapq subroutines are all for min-heap!
        duals=[],
        # the path taken through the search tree [(node, primal, dual)]
        track=[],
        # the total number of lp solver iterations
        lpiter=0,
        # the total number of bnb loop iterations
        iter=0,
        # own random number generator for tie resolution
        rng=default_rng(seed),
        # the root node is unlikely to be anything other than zero
        root=None,
        # the best (lowest) dual bound and node (typically the root, since
        #  its feasibility region is a super set for all sub-problems in the
        #  search tree, but may be another node due to numerical issues)
        dual_bound=DualBound(-np.inf, None),
        # statistics for variable pseudocosts
        pseudocosts=dict(
            lo=np.zeros(p.n),
            hi=np.zeros(p.n),
            n=np.zeros(p.n, int),
        ),
    )


def add(G: nx.DiGraph, p: MILP, *, errors: str = "raise") -> int:
    """Add a sub-problem to the search tree and the dual bound heap."""
    assert errors in ("raise", "ignore")

    # solve the relaxed LP problem w/o integrality constraints
    lp = lpsolve(p)
    G.graph["lpiter"] += lp.nit

    # The lp can be unbounded only at the root. Indeed, at no point does
    #  the BnB tree remove any constraint. Hence, if we detected an unbounded
    #  LP in a node of a sub-tree then it is also unbounded at the root of
    #  that sub-tree.
    if not lp.success and errors == "raise":
        raise RuntimeError(lp.message)

    # get the correct state and compute the fractional mask
    if lp.success:
        fractional = np.packbits(~is_feasible_int(lp.x, p))
        if fractional.any():
            status, mask = Status.OPEN, fractional

        else:
            status, mask = Status.FEASIBLE, None

    else:
        status, mask = Status.INFEASIBLE, None

    id = len(G)  # XXX max(G, default=0) + 1, or G.graph["n_nodes"] += 1
    # create the node with its MILP sub-problem, the relaxed lp solution, the
    #  mask of integer variables, that happen to have fractional values, the
    #  status, and the best-so-far integer-feasible solution in the sub-tree,
    #  which is initialized to own lp solution, when it is integer-feasible.
    best = lp if status == Status.FEASIBLE else None
    G.add_node(id, p=p, lp=lp, mask=mask, status=status, best=best)

    # the relaxation is INFEASIBLE, hence there is no lower bound to track
    #  and nothing to fathom anymore
    if not lp.success:
        return id

    # track the worst dual bound
    # XXX isn't this bound obtained right a the root by construction?
    dual = DualBound(-lp.fun, id)  # XXX -ve sign due to min-heap!
    if dual.val > G.graph["dual_bound"].val:
        G.graph["dual_bound"] = dual

    # we use min-heap with -ve dual values for easier pruning of OPEN nodes.
    # FEASIBLE nodes have already found the best possible integer-feasible
    #  solution, and do not require fathoming
    if status == Status.OPEN:
        heappush(G.graph["duals"], dual)

    elif G.graph["incumbent"].fun > lp.fun:
        assert status == Status.FEASIBLE

        # integer-feasible `lp.x` is the best possible solution in the current
        #  node's sub-problem, hence we update the global incumbent
        G.graph["incumbent"] = lp

    return id


def gap(G: nx.DiGraph) -> float:
    """Compute the primal-dual gap the way SCIP does it.

    Details
    -------
    SCIP's docs say that the gap is `+inf` if the primal and dual bounds have
    opposite signs, otherwise, it is
            `|primalbound - dualbound| / min(|primalbound|, |dualbound|)|`.
    """
    f_primal = G.graph["incumbent"].fun
    f_dual = -G.graph["dual_bound"].val
    if f_dual < 0 <= f_primal or f_primal < 0 <= f_dual:
        return float("inf")

    return abs(f_primal - f_dual) / min(abs(f_primal), abs(f_dual))


def prune(G: nx.DiGraph) -> None:
    """Mark nodes that certifiably cannot produce a better solution (primal < dual)."""
    # `.duals` is a min-heap of all nodes that have their -ve dual bounds lower
    #   than the global primal bound
    duals, incumbent = G.graph["duals"], G.graph["incumbent"]
    while duals and (incumbent.fun < -duals[0].val):
        # the node at the top of the heap, has a certificate that indicates that
        #  no solution in its sub-tree node is better than the current incumbent
        node = heappop(duals).node
        G.nodes[node]["status"] = Status.PRUNED
        # XXX if we wish to keep FEASIBLE status we need to
        #  slightly alter the logic of `bnb_update_incumbent`.


def branch(G: nx.DiGraph, node: int, by: int) -> tuple[int]:
    """Branch the search tree at the specified node."""
    data, children = G.nodes[node], []

    # XXX if we get a sub-problem P with a feasibility set, that
    #  is a proper subset of a MILP Q, with solution Q.x in P, or
    #  infeasible, then we could reuse Q and not consider P
    p, lp = data["p"], data["lp"]

    # partition the problem into non-overlapping sub-problems
    p_lo, p_hi = split(p, by, lp.x[by])
    for p, d in (p_lo, -1), (p_hi, +1):
        # create a new node and solve its relaxation
        leaf = add(G, p, errors="ignore")
        children.append(leaf)

        # link the node to the leaf (new or reused)
        G.add_edge(node, leaf, var=(by, d, lp.x[by]))

    # return the ids of the spawned children
    return tuple(children)
