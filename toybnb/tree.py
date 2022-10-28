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
        The sub-problem was eliminated due to sub-optimality, which
        was certified by the existende of an integer feasible incumbent,
        whose objective value was lower than the lp relxation bound.
    FEASIBLE:
        The lp relaxation of the node's sub-problem is integer-feasible and
        optimal, hence there is no reason to dive into its sub-tree. The branch
        is fathomed.
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
        # XXX we store bounds in `DualBound` nt-s with value of the
        #  opposite sign, since heapq subroutines are all for min-heap!
        duals=[],
        # the path taken through the search tree [(node, primal, dual)]
        track=[],
        # the best (lowest) dual bound and node (typically the root, since
        #  its feasibility region is a super set for all sub-problems in the
        #  search tree, but may be nother node due to numerical issues)
        dual_bound=DualBound(-np.inf, None),
        # the total number of lp solver iterations
        lpiter=0,
        # the total number of bnb loop iterations
        iter=0,
        # own random number generator for tie resolution
        rng=default_rng(seed),
        # statistics for variable pseudocosts
        pseudocosts=dict(
            lo=np.zeros(p.n),
            hi=np.zeros(p.n),
            n=np.zeros(p.n, int),
        ),
        root=None,
    )


def add(G: nx.DiGraph, p: MILP, *, errors: str = "raise") -> int:
    """Add a sub-problem to the search tree and the dual bound heap."""
    assert errors in ("raise", "ignore")

    # solve the realxed LP problem (w/o integrality)
    lp = lpsolve(p)
    G.graph["lpiter"] += lp.nit
    if not lp.success and errors == "raise":
        raise RuntimeError(lp.message)

    # assign the correct state to the node
    is_feas_int = is_feasible_int(lp.x, p)
    if lp.success and is_feas_int.all():
        status = Status.FEASIBLE

    elif lp.success:
        status = Status.OPEN

    else:
        status = Status.INFEASIBLE

    id = len(G)  # XXX max(G, default=0) + 1, or G.graph["n_nodes"] += 1
    # create the node with its MILP sub-problem, the relaxed lp solution, the
    #  packed mask of integer variables, that happen to have fractional values,
    #  the status, and the best-so-far integer-feasible solution in the subtree,
    #  which is initialized to the own lp solution when it is integer-feasible.
    mask = np.packbits(~is_feas_int)
    best = lp if status == Status.FEASIBLE else None
    G.add_node(id, p=p, lp=lp, mask=mask, status=status, best=best)

    # enqueue the node if LP was solved successfuly
    if lp.success:
        # node priority is -ve dual value (for easier pruning)
        dual = DualBound(-lp.fun, id)
        heappush(G.graph["duals"], dual)

        # track the best dual bound (mind the -ve sign!!!)
        if dual.val > G.graph["dual_bound"].val:
            G.graph["dual_bound"] = dual

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


def update_incumbent(G: nx.DiGraph, node: int) -> bool:
    """Checks of the current nodes's lp produced a better feasible solution."""
    status, lp = G.nodes[node]["status"], G.nodes[node]["lp"]

    # If the relaxed lp's value (dual) is better than the incumbent's
    #  objective value (primal) AND the lp's solution is integer feasible,
    #  then we can make it out new incumbent
    if status != Status.FEASIBLE or G.graph["incumbent"].fun < lp.fun:
        return False

    # XXX integer-feasible `lp.x` is the best solution in the current node's
    #  sub-problem, hence we can propagate it up the tree
    G.graph["incumbent"] = lp
    return True


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
