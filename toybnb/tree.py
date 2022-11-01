import numpy as np
import networkx as nx

from scipy.optimize import linprog, OptimizeResult
from numpy.random import default_rng
from heapq import heappush, heappop
from math import floor, ceil

from numpy import ndarray
from collections import namedtuple
from enum import Enum

from .milp import MILP, is_feasible_int


def build_optresult(
    x: ndarray = None,
    status: int = 1,
    message: str = "",
    fun: float = np.inf,
    nit: int = 0,
    **ignore,
) -> OptimizeResult:
    """Build an OptimizeResult with only the essential fields."""
    return OptimizeResult(
        # the solution of the optimization
        x=x,
        # whether or not the optimizer exited successfully
        success=status == 0,
        # termination status code of the optimizer
        # 0 success, 1 iteration/time limit, 2 infeasible, 3 unbounded, 4 other
        status=status,
        # description of the cause of the termination
        message=message,
        # the value of the objective function
        fun=fun,
        # number of iterations performed by the optimizer
        nit=nit,
    )


def lpsolve(p: MILP) -> OptimizeResult:
    """Drop the integrality constraints and solve the relaxed LP."""
    # use scipy's linprog to solve the relaxed problem
    #     $\min_x \{ c^\top x : A x \leq b, x \in [l, u] \}$
    # XXX linprog does a sizeable amount of pre-processing and cleaning before
    #  actually solving. we could benefit from calling the solver directly.
    # XXX bounds can be in an `N x 2` array with `\pm\infty` bounds.
    #  see [_clean_inputs](scipy/optimize/_linprog_util.py#L390-452)
    lp = linprog(p.c, p.A_ub, p.b_ub, p.A_eq, p.b_eq, p.bounds, method="highs")
    if lp.success:
        # `.x` is $x^*$ the solution and `.fun` is $c^\top x^*$
        return build_optresult(**lp)

    # the data in OptimizeResult in case of an error status varies depending
    #  on the method, thus we construct our own result
    return build_optresult(
        x=None,
        fun=-np.inf if lp.status == 3 else +np.inf,
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
            \bigl(\lfloor x_j\rfloor, \lceil x_j\rceil \bigr)
                \times \mathbb{R}^{n-1}
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
        A MILP sub-problem has the lp lower bound, which is better, than the
        current global incumbent solution. The node still has some branching
        options left to explore. OPEN nodes may ARBITRARILY change their status
        to CLOSED or PRUNED.
    CLOSED:
        All branching options offered by this node have been exhausted, but its
        sub-tree has not necessarily been fathomed. A CLOSED node may only
        change its status to PRUNED.
    PRUNED:
        The node had to be eliminated due to certifiable sub-optimality. During
        the search a non overlapping sub-problem produced an integer-feasible
        solution with a LOWER objective value, than this node's lp relaxation,
        which, by design, bounds any integer-feasible solution in its region.
        A PRUNED node MUST stay pruned forever.
    FEASIBLE:
        The lp relaxation of the node's sub-problem is integer-feasible and
        optimal, hence there is no reason to dive into its sub-tree. The branch
        is fathomed.
        A FEASIBLE node may only change its status to PRUNED. Feasible nodes can
        reside in a pruned branch, in which case it is also considered pruned.
    INFEASIBLE:
        The relaxed lp has a degenerate feasibility region, meaning that node's
        the MILP sub-problem is infeasible. Again, the sub-tree is fathomed.
        An INFEASIBLE node never changes its status.

    Closed nodes
    ------------
    If the node's lp solution is integer infeasible, then the subset of the
    node's feasibility region NOT COVERED by any axis-aligned binary split on
    fractional variables is excluded from all sub-problems in the sub-tree. This
    region is an open rectangle, possibly unbounded, with at least one side
    being an open interval between CONSECUTIVE integers. Thus, this region can
    be safely excluded from consideration, since is cannot contain any integer
    feasible solutions to the node's problem.

    A sub-problem in a partition of a binary split either may be immediately
    infeasible (empty feasibility set), or may have produced an integer feasible
    candidate after diving after exhausting all branching options. Although the
    current node's lp lower bound might still have a large gap with respect to
    the primal bound, we can safely mark this node as fathomed.

    CLOSED status does not mean that the entire sub-tree of a node has been
    fathomed, only that this node's local branching options have all been
    explored. For example, it is possible that an OPEN node while still siting
    in the `duals` heap with lp dual < primal, may in the same iteration change
    its status to CLOSED, have one immediate integer-feasible child improve
    the incumbent, making the node eligible for pruning, yet have another child
    completely unfathomed.

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
    return nx.DiGraph(
        None,
        # the original primal linear problem
        p=p,
        # the best integer feasible solution found so far
        incumbent=build_optresult(x=None, fun=np.inf, status=1, message=""),
        # the queue for sub-problem prioritization
        queue=[],  # deque([]),
        # the max-heap of OPEN nodes ordered by the lp bound for faster pruning
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
        # the worst (lowest) dual bound and node (typically the root, since
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


def add(G: nx.DiGraph, p: MILP, **attrs: dict) -> int:
    """Add a new MILP sub-problem to the search tree."""
    # solve the relaxed LP problem w/o integrality constraints
    lp = lpsolve(p)
    G.graph["lpiter"] += lp.nit

    # get the correct state and compute the fractional mask
    if lp.success:
        fractional = np.packbits(~is_feasible_int(lp.x, p))
        if fractional.any():
            # If a MILP sub-problem has an integer-INfeasible lp solution, then
            #  its feasibility region has to be searched further
            status, mask = Status.OPEN, fractional

        else:
            # If an optimal lp solution is also integer-feasible, then `lp.x`
            #  is the BEST possible solution to this nodes MILP sub-problem.
            status, mask = Status.FEASIBLE, None

    else:
        # The lp relaxation of the MILP is either infeasible or unbounded. By
        #  construction, unboundedness can only happen at the root. Indeed, BnB
        #  search progressively shrinks the feasibility set, which means that
        #  if we detect an unbounded node then this node's parent is also
        #  unbounded. Therefore we only have to worry about the root.
        status, mask = Status.INFEASIBLE, None

    id = len(G)  # XXX max(G, default=0) + 1, or G.graph["n_nodes"] += 1
    # create the node with its MILP sub-problem, the relaxed lp solution, the
    #  mask of integer variables, that happen to have fractional values, the
    #  status, and the best-so-far integer-feasible solution in the sub-tree,
    #  which is initialized to own lp solution, when it is integer-feasible.
    best = lp if status == Status.FEASIBLE else None
    G.add_node(id, p=p, lp=lp, mask=mask, status=status, best=best, **attrs)

    # the relaxation is INFEASIBLE, hence there is no lower bound to track
    #  and nothing to fathom anymore
    if not lp.success:
        return id

    # We store the lp bound in `DualBound` with the OPPOSITE sign, since we'll
    #  be using it to prune certifiably sub-optimal nodes with min-heap procs
    #  from `heapq`
    dual = DualBound(-lp.fun, id)

    # track the worst lp lower bound
    # XXX isn't this bound obtained right at the root by construction?
    if dual.val > G.graph["dual_bound"].val:
        G.graph["dual_bound"] = dual

    # the `duals` heap tracks only those nodes, the MILP sub-problem of which
    #  still needs their sub-tree explored.
    if status == Status.OPEN:
        heappush(G.graph["duals"], dual)

    # see if the integer-feasible solution may update the global incumbent
    elif G.graph["incumbent"].fun > lp.fun:
        assert status == Status.FEASIBLE
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
    """Mark OPEN nodes that certifiably CANNOT produce a better solution."""
    duals, incumbent, nodes = G.graph["duals"], G.graph["incumbent"], G.nodes

    # the node at the top of the heap, has a certificate that indicates that
    #  no solution in its sub-tree node is better than the current incumbent
    while duals and (incumbent.fun < -duals[0].val):
        node = heappop(duals).node

        # we can only see OPEN or CLOSED nodes in this heap, since INFEASIBLE
        #  and FEASIBLE are never added to the heap, and PRUNED never reenter.
        assert nodes[node]["status"] in (Status.OPEN, Status.CLOSED)
        nodes[node]["status"] = Status.PRUNED


def backup(G: nx.DiGraph, node: int) -> None:
    """Propagate this node's integer-feasible solution up the tree."""
    data = G.nodes[node]
    if data["status"] != Status.FEASIBLE:
        return

    # the best integer-feasible solutions live in a min-tree: the parent's
    #  solution is guaranteed to be not worse than a solution of any child
    best = data["best"]
    while G.pred[node]:
        # get the parent and see if the min-tree needs fixing
        node = next(iter(G.pred[node]))
        data = G.nodes[node]
        if data["best"] is not None and data["best"].fun <= best.fun:
            return

        # update the parent's best-so-far solution with the child's
        data["best"] = best


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
        leaf = add(G, p, depth=data["depth"] + 1)
        children.append(leaf)

        # link the node to the leaf (new or reused)
        G.add_edge(node, leaf, var=(by, d, lp.x[by]))

        # propagate the feasible solution up the branch
        backup(G, leaf)

    # return the ids of the spawned children
    return tuple(children)
