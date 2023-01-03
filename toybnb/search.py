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


def search(
    p: MILP,
    nodesel: callable,
    branchrule: callable,
    *,
    gap: float = 0.0,
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
        for j in pbar:
            # monitor bnb search progress
            f_gap = bnb.gap(T)
            if verbose > 0:
                pbar.set_postfix_str(f"rtol {f_gap:.4%} {len(duals)} {len(queue)}")

            # terminate if the target primal-dual gap is surpassed or the node
            #  budget is exceeded
            # XXX should we delegate these checks to nodesel? On the one hand,
            #  separation of responsibilites, on the other hand, nodsel has
            #  more power: it selects nodes or quits. The nature of termination
            #  is different, though.
            if f_gap <= gap or len(T) > n_nodes:
                break

            # `nodesel` gets the next OPEN node to visit, or raises IndexError
            node = nodesel(T, *children)

            # the picked node's lp solution is expected to be integer-infeasible,
            #  and still have branching options left, i.e. OPEN.
            data = nodes[node]
            assert data["status"] == Status.OPEN

            # pick a branching rule and with it a variable for branching
            # XXX this should raise IndexError if no variable can be picked
            j = branchrule(T, node)

            # mark the node as CLOSED, since there is no reason for multiple branchings
            data["status"] = Status.CLOSED

            # sprout and schedule the shoots, and reschedule the node
            #  itself, since it may have other variables to explore
            children = bnb.branch(T, node, j)

            # add the focus node to the track after a successful branching
            track.append((node, T.graph["incumbent"].fun, data["lp"].fun, float("nan")))
            # XXX the track may only have OPEN nodes, CLOSED nodes, or
            #  PRUNED nodes when lp bound > primal evantually

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
