import networkx as nx

from tqdm import tqdm
from itertools import count

from . import tree as bnb
from .milp import MILP
from .tree import Status


def nodesel_dfs(G: nx.DiGraph, *reschedule: int) -> int:
    """A simple node selector which prioritizes depth.

    Parameters
    ----------
    G: nx.DiGraph
        The  bnb search tree.

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
    dq, nodes = G.graph["queue"], G.nodes
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
    seed: int = None,
    verbose: int = 0,
) -> nx.DiGraph:
    """Poor-man's branch and bound algorithm."""
    n_nodes = float("inf") if n_nodes is None else n_nodes

    # initialize the bnb tree
    tree = bnb.init(p, seed=seed)

    # localize certain variables
    queue = tree.graph["queue"]
    nodes = tree.nodes
    track = tree.graph["track"]
    duals = tree.graph["duals"]
    try:
        pbar = tqdm(count(), ncols=70, disable=verbose < 1)

        # create the root node
        tree.graph["root"], _ = bnb.add(tree, p, depth=0)
        reschedule = [tree.graph["root"]]

        # start the bnb loop
        for j in pbar:
            # monitor bnb search progress
            f_gap = bnb.gap(tree)
            if verbose > 0:
                pbar.set_postfix_str(f"rtol {f_gap:.4%} {len(duals)} {len(queue)}")

            # terminate if the target primal-dual gap is surpassed or the node
            #  budget is exceeded
            # XXX should we delegate these checks to nodesel? On the one hand,
            #  separation of responsibilites, on the other hand, nodsel has
            #  more power: it selects nodes or quits. The nature of termination
            #  is different, though.
            if f_gap <= gap or len(tree) > n_nodes:
                break

            # `nodesel` gets the next OPEN node to visit, or raises IndexError
            node = nodesel(tree, *reschedule)
            reschedule.clear()

            # the picked node's lp solution is expected to be integer-infeasible,
            #  and still have branching options left, i.e. OPEN.
            data = nodes[node]
            assert data["status"] == Status.OPEN

            # try branching at the node, marking is CLOSED if impossible
            try:
                # pick a branching rule and with it a variable for branching
                # XXX this should raise IndexError if no variable can be picked
                j = branchrule(tree, node)

                # sprout and schedule the shoots, and reschedule the node
                #  itself, since it may have other variables to explore
                children = bnb.branch(tree, node, j)
                reschedule.append(node)
                reschedule.extend(children)

                # add the focus node to the track after a successful branching
                track.append((node, tree.graph["incumbent"].fun, data["lp"].fun))
                # XXX the track may only have OPEN nodes, CLOSED nodes, or
                #  PRUNED nodes when lp bound > primal evantually

            except IndexError:
                # mark the node as CLOSED, since no variable can be branched on
                data["status"] = Status.CLOSED

            # the tree has changed: prune certifiably sub-optimal nodes
            # XXX `duals` NEVER runs out of nodes before `nodesel` raises
            bnb.prune(tree)

            tree.graph["iter"] += 1

    except (IndexError, KeyboardInterrupt):
        pass

    finally:
        pbar.close()

    # unless stopped early, at this point `duals` should be empty
    return tree
