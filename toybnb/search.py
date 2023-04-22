from itertools import count

import networkx as nx
from tqdm import tqdm

from . import tree as bnb
from .milp import MILP
from .tree import Status


def nodesel_dfs(T: nx.DiGraph, *reschedule: int) -> int:
    """A simple node selector which prioritizes depth.

    Parameters
    ----------
    T: nx.DiGraph
        The bnb search tree.

    reschedule: int
        The list of nodes to put back into the queue. It is either empty, or
        it a list with the node itself at the first position, followed by its
        immediate children.

    Returns
    -------
    node: int
        The next OPEN node to visit. The `IndexError` is raised if nothing
        can be proposed.
    """
    dq, nodes = T.graph["queue"], T.nodes
    for node in reschedule:
        # a rescheduled node may be FEASIBLE, INFEASIBLE, OPEN, or PRUNED
        assert nodes[node]["status"] != Status.CLOSED
        # XXX `reschedule` contains the just visited node and its children
        #  created by branching. These cannot be CLOSED nodes: either OPEN
        #  with branching options left, as determined by the `branchrule`
        #  __not__ raising IndexError, FEASIBLE or INFEASIBLE children or
        #  even PRUNED. The child cannot prune its parent, but can affect
        #  its sibling: the incumbent is lower bounded by the child's and
        #  the parent's relaxed lp bounds, but the sibling's lp bound came
        #  out too high.

        # we schedule only OPEN nodes
        if nodes[node]["status"] == Status.OPEN:
            dq.append(node)

    while dq:
        node = dq.pop()

        # node's status OPEN may have changed to PRUNED or CLOSED
        if nodes[node]["status"] == Status.OPEN:
            return node

    raise IndexError


def gap(T: nx.DiGraph) -> float:
    """Compute the primal-dual gap the way SCIP does it.

    Details
    -------
    SCIP's docs say that the gap is `+inf` if the primal and dual bounds have
    opposite signs, otherwise, it is
            `|primalbound - dualbound| / min(|primalbound|, |dualbound|)|`.
    """
    f_primal = T.graph["incumbent"].fun
    f_dual = T.graph["dual_bound"]
    if f_dual < 0 <= f_primal or f_primal < 0 <= f_dual:
        return float("inf")

    return abs(f_primal - f_dual) / min(abs(f_primal), abs(f_dual))


def search(
    p: MILP,
    nodesel: callable,
    branchrule: callable,
    *,
    f_gap: float = None,
    n_nodes: float = None,
    verbose: int = 0,
) -> nx.DiGraph:
    """Poor-man's branch and bound algorithm."""
    n_nodes = float("inf") if n_nodes is None else n_nodes

    # initialize the bnb tree
    T = bnb.init(
        p,
        # the root node is unlikely to be anything other than zero
        root=None,
        # the total number of bnb loop iterations
        iter=0,
        # the path taken through the search tree [(node, primal, dual)]
        track=[],
        # the queue for sub-problem prioritization
        queue=[],  # deque([]),
        # the worst (lowest) dual bound among the open nodes
        dual_bound=float("inf"),
    )

    # localize certain variables
    nodes = T.nodes
    duals = T.graph["duals"]
    track = T.graph["track"]
    queue = T.graph["queue"]
    try:
        pbar = tqdm(count(), ncols=70, disable=verbose < 1)

        # create the root node
        T.graph["root"], _ = bnb.add(T, p, depth=0)
        children = [T.graph["root"]]

        # start the bnb loop
        root = T.graph["root"]
        for j in pbar:
            # monitor bnb search progress
            f_current_gap = gap(T)
            if verbose > 0:
                pbar.set_postfix_str(
                    f"rtol {f_current_gap:.4%} {len(duals)} {len(queue)}"
                )

            # terminate if the target primal-dual gap is surpassed or the node
            #  budget is exceeded
            # XXX should we delegate these checks to nodesel? On the one hand,
            #  separation of responsibilites, on the other hand, nodsel has
            #  more power: it selects nodes or quits. The nature of termination
            #  is different, though.
            if (f_gap is not None and f_current_gap <= f_gap) or len(T) > n_nodes:
                break

            # `nodesel` gets the next OPEN node to visit, or raises IndexError
            node = nodesel(T, *children)

            # the picked node's lp solution is expected to be integer-infeasible,
            #  and still have branching options left, i.e. OPEN.
            data = nodes[node]
            assert data["status"] == Status.OPEN

            # pick a branching rule and with it a variable for branching
            # XXX rules raise IndexError if no variable can be picked. We
            #  reraise as RuntimeError
            try:
                j = branchrule(T, node)

            except IndexError as e:
                raise RuntimeError(e) from None

            # mark the node as CLOSED, since there is no reason for multiple branchings
            data["status"] = Status.CLOSED

            # sprout and schedule the shoots, and reschedule the node itself,
            #  since it may have other variables to explore
            children = bnb.branch(T, node, j)

            # add the focus node to the track after a successful branching
            track.append(
                (node, T.graph["incumbent"].fun, data["lp"].fun, T.graph["dual_bound"])
            )
            # XXX the track may only have OPEN nodes, CLOSED nodes, or PRUNED
            #  nodes when lp bound > primal evantually

            # propagate the worst dual bound, since the focus is no longer open
            # XXX Each node has its lp bound \breve{f}_n. By construction, its
            #  childrens' bounds \breve{f}_j \geq \breve{f}_n for j \in T_n. Thus
            #  if the focus is open, then there is no need to check its children
            #    b_n = \breve{f}_n                  if not `CLOSED`
            #          \min\{b_j \colon j \in T_n\} if `CLOSED` and has children
            #          +\infty                      if `CLOSED` and childless
            while data["status"] == Status.CLOSED:
                # XXX the worst lower bound of the a CLOSED node is the lowest
                #  of its childrens' worst lower bounds
                data["lb"] = min(float("inf"), *(nodes[c]["lb"] for c in T[node]))
                if not T.pred[node]:
                    break

                # get the parent and see if the min-tree needs fixing
                node = next(iter(T.pred[node]))
                data = nodes[node]

            # read the worst lower bound off the root node's lb
            T.graph["dual_bound"] = nodes[root]["lb"]

            # the tree has changed: prune certifiably sub-optimal nodes
            # XXX `duals` NEVER runs out of nodes before `nodesel` raises
            bnb.prune(T)

            T.graph["iter"] += 1

    except (IndexError, KeyboardInterrupt):
        pass

    finally:
        pbar.close()

    # unless stopped early, at this point `duals` should be empty
    return T
