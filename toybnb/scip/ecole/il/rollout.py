from typing import Union, Iterable, Callable
from time import perf_counter

from queue import Queue
from threading import Thread, Event

from ecole.scip import Model
from ecole.environment import Branching
from .data import Observation
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


def generator(feed: Iterable, signal: Event) -> Iterable:
    it = iter(feed)
    try:
        while not signal.is_set():
            yield next(it)

    except StopIteration:
        pass


def t_problem_feed(it: Iterable, put: Callable) -> None:
    try:
        for j, item in enumerate(it):
            put((True, j, item))

    except BaseException as e:
        put((False, -1, e))

    finally:
        put(None)


def t_rollout_worker(
    rk: int,
    factory: Callable,
    branchrule: Callable,
    get: Callable,
    put: Callable,
    stop: Callable[[], bool] = None,
) -> None:
    try:
        # create the branching env
        env = factory()

        # keep solving problems until a stop order (None)
        for success, j, p in iter(get, None):
            if not success:
                raise p from None

            # dispatch each step from rollout
            for item in rollout(p, env, branchrule, {}, stop=stop):
                put((True, rk, j, item))

            # check the stopping condition
            if stop():
                break

    except BaseException as e:
        put((False, rk, j, e))

    finally:
        put(None)


def t_evaluate_worker(
    rk: int,
    factory: Callable,
    branchrules: Callable,
    get: Callable,
    put: Callable,
    stop: Callable[[], bool] = None,
) -> None:
    try:
        # create the branching env
        env = factory()

        # keep solving problems until a stop order (None)
        for success, j, p in iter(get, None):
            if not success:
                raise p from None

            # do a rollout on this instance with each branchrule
            out = {k: evaluate(p, env, rule, {}) for k, rule in branchrules.items()}

            # forward the evaluation results
            put((True, rk, j, out))

            # check the stopping condition
            if stop():
                break

    except BaseException as e:
        put((False, rk, j, e))

    finally:
        put(None)


def pool_rollout(
    feed: Iterable,
    factories: tuple[Callable],
    branchrule: BranchRule,
    *,
    maxsize: int = 128,
) -> Iterable:
    worker = t_evaluate_worker if isinstance(branchrule, dict) else t_rollout_worker

    # the finish signal controls the upstream problem generator
    finish, problems, results = Event(), Queue(maxsize), Queue(maxsize)

    # the upstream generator puts the problems into the common shared queue
    it = generator(feed, finish)
    threads = [Thread(target=t_problem_feed, args=(it, problems.put))]

    # the downstream workers rollout on the received problems
    for rk, factory in enumerate(factories):
        args = rk, factory, branchrule, problems.get, results.put, finish.is_set
        threads.append(Thread(target=worker, args=args, daemon=True))

    try:
        for t in threads:
            t.start()

        for success, rk, j, result in iter(results.get, None):  # noqa: B007
            if not success:
                raise result from None

            # the main thread yields results from the rollout output queue
            yield result

    finally:
        # terminate the feeder thread and drain its outqueue (to avoid
        #  blocking during stop order issuing)
        finish.set()
        while not problems.empty():
            problems.get_nowait()

        # first issue a stop order to each thread and then join in a separate loop
        for _ in threads:
            problems.put(None)

        for t in threads:
            t.join()
