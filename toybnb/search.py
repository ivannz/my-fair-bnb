import networkx as nx

from tqdm import tqdm
from itertools import count

from . import tree as bnb
from .milp import MILP
from .tree import Status


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
        # create the root node
        tree.graph["root"] = bnb.add(tree, p, depth=0)
        reschedule = [tree.graph["root"]]

        # start the bnb loop
        pbar = tqdm(count(), ncols=70, disable=verbose < 1)
        for j in pbar:
            # monitor bnb search progress
            f_gap = bnb.gap(tree)
            if verbose > 0:
                pbar.set_postfix_str(f"rtol {f_gap:.4%} {len(duals)} {len(queue)}")

            # terminate if the target primal-dual gap is surpassed or the node
            #  budget is exceeded
            if f_gap <= gap or len(tree) > n_nodes:
                break

            tree.graph["iter"] += 1

            # get the next OPEN node to visit
            # XXX raises IndexError,if no node can be selected
            node = nodesel(tree, *reschedule)
            reschedule.clear()

            # the node's lp solution is integer infeasible, and it might have
            #  branching options left -- add it to the track anyway
            data = nodes[node]
            assert data["status"] == Status.OPEN
            track.append((node, tree.graph["incumbent"].fun, data["lp"].fun))
            # XXX the track NEVER contains any INFEASIBLE nodes: it may have
            #  OPEN nodes, if early stopped, PRUNED nodes, with dual > primal,
            #  and CLOSED nodes. Due to the above assert it may not contain
            #  FEASIBLE nodes, although the nodesel could return such nodes.

            # try branching at the node, marking is CLOSED if impossible
            try:
                # pick a branching rule and with it a variable for branching
                # XXX this should raise IndexError if no variable can be picked
                j = branchrule(tree, node)

                # sprout and schedule the shoots, and reschedule the node
                #  itself, since it may have other variables to explore
                children = bnb.branch(tree, node, j)
                reschedule.extend(children)
                reschedule.append(node)

            except IndexError:
                # mark the node as CLOSED, since no variable can be branched on
                data["status"] = Status.CLOSED

            # the tree has changed: prune certifiably sub-optimal nodes
            # XXX `duals` NEVER runs out of nodes before `nodesel` raises
            bnb.prune(tree)

    except (IndexError, KeyboardInterrupt):
        pass

    finally:
        pbar.close()

    # unless stopped early, at this point `duals` should be empty
    return tree
