from typing import Union, Iterable, Callable
from time import perf_counter, process_time

from ecole.scip import Model
from ecole.environment import Branching
from .data import Observation
from .brancher import BranchRule


def check_sigint(m: Model) -> None:
    """Manually check if SCIP caught a sigint.

    SCIP has a nasty habit of intercepting and muffling Keyboard Interrupts.
    A workaround is to check if the SCIP's model's status indicates termination
    due to a sigint.
    """
    if m.as_pyscipopt().getStatus() == "userinterrupt":
        raise KeyboardInterrupt from None


def rollout(
    p: Union[Model, str],
    env: Branching,
    branchrule: BranchRule,
    kwargs: dict = None,
    stop: Callable[[], bool] = None,
) -> Iterable:
    """A generator of observation-action-reward data from the nodes of SCIP's BnB
    search, at which a branching decision was made.
    """

    stop = stop if callable(stop) else (lambda: False)
    kwargs = {} if kwargs is None else kwargs
    do_branch = branchrule(env, **kwargs)

    wt, pt = perf_counter(), process_time()

    # do a rollout, sending the results into the buffer
    n_steps = 0
    obs, act_set, rew, fin, nfo = env.reset(p)
    while not fin and not stop():
        check_sigint(env.model)

        # the action set should be treated as a part of the observation
        obs_ = Observation(obs, act_set)

        # query the expert and branch
        act_ = do_branch(obs_)
        obs, act_set, rew, fin, _ = env.step(act_)  # t -->> t+1
        n_steps += 1

        # send out the `x_{t-1}, a_{t-1}, r_t`
        yield obs_, act_, rew  # XXX no underscore in `rew`!
        # XXX SCIP has a complex node selection strategy, which even
        #  when set to prioritize DFS, still may switch no arbitrary
        #  node after branching. For the purpose of this experiment
        #  we make the worst-case assumption about the transition
        #  funciton that the next focus node is not at all related
        #  to the a prior branching decision. The only assumption is
        #  that the returned reward reflects the quality of the branching
        #  decision

    check_sigint(env.model)

    # no need to yield anything on fin=True, since ecole's terminal
    #  observation is None
    wt, pt = perf_counter() - wt, process_time() - pt
    m = env.model.as_pyscipopt()
    return dict(
        n_requests=n_steps,
        n_nodes=m.getNNodes(),
        n_lpiter=m.getNLPIterations(),
        n_lps=m.getNLPs(),
        f_soltime=m.getSolvingTime(),
        f_gap=m.getGap(),
        s_status=m.getStatus(),
        f_walltime=wt,
        f_proctime=pt,
    )


def evaluate(
    p: Model, env: Branching, branchrule: BranchRule, kwargs: dict = None
) -> dict[str, ...]:
    """Get the final metrics after branchrule rollout"""
    try:
        # use while-loop to capture the return value from the generator
        it = rollout(p, env, branchrule, kwargs)
        while True:
            next(it)

    except StopIteration as e:
        return e.value
