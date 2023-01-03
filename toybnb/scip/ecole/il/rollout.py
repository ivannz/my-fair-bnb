from typing import Union, Iterable, Callable
from time import perf_counter

from queue import Queue
from threading import Thread, Event

from ecole.scip import Model
from ecole.environment import Branching
from .env import Observation
from .brancher import BranchRule


def maybe_raise_sigint(m: Model) -> None:
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

    f_wtime = perf_counter()

    # do a rollout, sending the results into the buffer
    n_steps = 0
    obs, act_set, rew, fin, nfo = env.reset(p)
    while not fin and not stop():
        maybe_raise_sigint(env.model)

        # the action set should be treated as a part of the observation
        obs_ = Observation(obs, act_set)

        # query the expert and branch
        act_ = do_branch(obs_)
        obs, act_set, rew, fin, _ = env.step(act_)  # t -->> t+1
        n_steps += 1

        # send out the `x_{t-1}, a_{t-1}, r_t`
        yield obs_, act_, rew  # XXX no underscore in `rew`!
        # XXX SCIP has a complex node selection strategy, which even when set to
        #  prioritize DFS, still may switch no arbitrary node after branching.
        #  For the purpose of this experiment we make the worst-case assumption
        #  about the transition funciton that the next focus node is not at all
        #  related to a prior branching decision. The only assumption is that
        #  the returned reward reflects the quality of the branching decision

    f_wtime = perf_counter() - f_wtime

    maybe_raise_sigint(env.model)

    # no need to yield anything on `fin=True`, since ecole's terminal obs is None
    m = env.model.as_pyscipopt()
    return dict(
        n_interactions=n_steps,
        n_nodes=m.getNNodes(),
        n_lpiter=m.getNLPIterations(),
        n_lps=m.getNLPs(),
        f_soltime=m.getSolvingTime(),
        f_gap=m.getGap(),
        s_status=m.getStatus(),
        f_walltime=f_wtime,
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
        # the exception payload contains the post-rollout tree stats
        return e.value


def t_problem_feed(feed: Iterable, put: Callable, stop: Callable[[], bool]) -> None:
    try:
        it = enumerate(feed)
        while not stop():
            j, item = next(it)
            put((True, j, item))

    except StopIteration:
        pass

    except BaseException as e:
        put((False, -1, e))

    finally:
        put(None)


def t_rollout_worker(
    rk: int,
    factory: Callable,
    get: Callable,
    put: Callable,
    stop: Callable[[], bool] = None,
    **rules: dict[str, BranchRule],
) -> None:
    stop = stop if callable(stop) else (lambda: False)

    try:
        # create the branching env
        env = factory()

        # keep solving problems until a stop order (None)
        for success, j, p in iter(get, None):
            if not success:
                raise p

            # we can rollout on several branchrules, one after another
            for name, rule in rules.items():
                # dispatch each step from rollout
                for item in rollout(p, env, rule, {}, stop=stop):
                    put((True, rk, j, (name, item)))

                # check the stopping condition
                if stop():
                    return

    except BaseException as e:
        put((False, rk, j, e))

    finally:
        put(None)


def t_evaluate_worker(
    rk: int,
    factory: Callable,
    get: Callable,
    put: Callable,
    stop: Callable[[], bool] = None,
    **rules: dict[str, BranchRule],
) -> None:
    stop = stop if callable(stop) else (lambda: False)

    try:
        # create the branching env
        env = factory()

        # keep solving problems until a stop order (None)
        for success, j, p in iter(get, None):
            if not success:
                raise p

            # do a rollout on this instance with each branchrule
            out = {k: evaluate(p, env, rule, {}) for k, rule in rules.items()}

            # forward the evaluation results
            put((True, rk, j, out))

            # check the stopping condition
            if stop():
                return

    except BaseException as e:
        put((False, rk, j, e))

    finally:
        put(None)


def pool_rollout(
    feed: Iterable,
    factories: tuple[Callable],
    worker: Callable = t_rollout_worker,
    **rules: dict[str, BranchRule],
) -> Iterable:
    """Rollout the specified branchrule in Ecole's branching environment"""
    assert worker is t_rollout_worker or worker is t_evaluate_worker

    # the finish signal controls the upstream problem generator
    # XXX limit the size of the `results` queue so that the workers would be
    #  eventually suspended if the generator is no longer requested data from
    finish, problems, results = Event(), Queue(128), Queue(1024)

    # the upstream generator puts the problems into the common shared queue
    threads = [Thread(target=t_problem_feed, args=(feed, problems.put, finish.is_set))]

    # the downstream workers rollout on the received problems
    for rk, factory in enumerate(factories):
        args = rk, factory, problems.get, results.put, finish.is_set
        threads.append(Thread(target=worker, args=args, kwargs=rules, daemon=True))

    try:
        for t in threads:
            t.start()

        for success, rk, j, result in iter(results.get, None):  # noqa: B007
            if not success:
                raise result

            # the main thread yields results from the rollout output queue
            yield result

    finally:
        # terminate the feeder thread and drain its outqueue to unblock it
        finish.set()
        while not problems.empty():
            problems.get_nowait()

        # issue a stop order to each worker thread, drain their outqueue to unblock
        for _ in threads:
            problems.put(None)

        while not results.empty():
            results.get_nowait()

        # join the feeder and the workers threads
        for t in threads:
            t.join()
