from threading import Condition, Thread, Lock
from time import sleep


class Coroutine(Thread):
    """My own coroutine for control inversion.

    Can use `return` and `raise` in the target, but have to call `.co_yield`
    instead of `yield`.
    """

    nothing: object = object()  # a sentinel object

    class OutOfSequenceError(RuntimeError):
        pass

    def __init__(self, target, name=None, args=(), kwargs=None, daemon=None):
        super().__init__(None, target, name, args, kwargs, daemon=daemon)

        # cv is used for lock-step synchronisation, while its lock is the mutex
        # """All methods of `Lock` are executed atomically."""
        # https://docs.python.org/3/library/threading.html#lock-objects
        self.cv = Condition(Lock())
        self.exception_, self.value_ = None, self.nothing
        self.co_is_running, self.co_is_suspended = False, False
        self.co_has_started, self.co_is_finished = False, False

    def maybe_raise(self, clear: bool) -> ...:
        """Either return the value of raise the pending exception."""
        try:
            if self.exception_ is not None:
                raise self.exception_

            if self.value_ is not self.nothing:
                return self.value_

            raise RuntimeError

        finally:
            # reset the value, but not the exception, unless in their thread.
            #  This can be used to tell if the exception at `.wait` originated
            #  from `co_raise` or from elsewhere.
            self.value_ = self.nothing
            if clear:
                self.exception_ = None

    def co_enter(self) -> bool:
        """Wait until our time-slice signal then suspend `.acquire` in their
        thread until after `.co_leave` in ours.
        """
        self.cv.acquire()
        return self.cv.wait_for(lambda: self.co_is_running)
        # XXX `.co_enter` is followed by `.co_leave` in our thread

    def co_leave(self) -> None:
        """Give up our time-slice, and allow their waiting `.acquire` to resume."""
        self.co_is_running = False
        self.cv.notify(1)
        self.cv.release()

    def co_yield(self, value: ... = None, stall: float = 0.0) -> ...:
        """Unsuspend them at `.wait`, wait for their `.resume`"""
        self.exception_, self.value_ = None, value
        self.co_leave()

        if stall > 0:
            sleep(stall)

        self.co_enter()
        return self.maybe_raise(True)

    def co_raise(self, exception: BaseException) -> None:
        """Raise an exception at `.wait`"""
        # ignore subsequent calls to co-raise
        if self.co_is_finished:
            return

        self.co_is_finished = True
        # XXX if we raise at them, then we are sure that we cannot continue
        #  otherwise we wouldn't have raised at them in the first place.
        self.exception_, self.value_ = exception, self.nothing
        self.co_leave()

    def co_return(self, value: ... = None) -> None:
        # they return by raising StopIteration at our `.wait`
        exception = StopIteration if value is None else StopIteration(value)
        return self.co_raise(exception)

    def run(self):
        self.co_has_started = self.co_is_running = True
        try:
            self.co_enter()
            self.co_return(self._target(*self._args, **self._kwargs))

        except BaseException as e:
            self.co_raise(e)

        finally:
            del self._target, self._args, self._kwargs

    def enter(self) -> None:
        """Wait for our time-slice signal, then postpone `.co_enter` in their
        thread until `.release` in our thread.
        """
        # XXX `.acquire` is followed by `.release` in our thread
        if self.co_is_suspended:
            raise self.OutOfSequenceError()

        self.co_is_suspended = self.cv.acquire()  # atomic, but wait can be interrupted
        if not self.co_is_finished:
            self.cv.wait_for(lambda: not self.co_is_running)
        # XXX `co_is_suspended` can end up being False only if `.acquire`
        # in `.enter` were interrupted, but never when `.wait_for`

    def leave(self) -> None:
        """our `.release` is coupled to their `.co_enter`"""
        assert self.co_is_suspended

        try:
            # there is no-one to notify if they are not running
            if not self.co_is_finished:
                self.co_is_running = True
                self.cv.notify(1)

        finally:
            self.co_is_suspended = False
            self.cv.release()

    def wait(self) -> ...:
        """Wait for their `.co_yield`."""
        # abort, if they have not started (we use Thread's private flag)
        if not self.co_has_started:
            raise self.OutOfSequenceError("Call `.start` to launch the coroutine.")

        # If we own the time-slice, then we know that they are inside `co_yield
        #  and are blocked. if they own, then we block only if we need the
        #  time-slice.
        self.enter()
        if not self.co_is_finished:
            return self.maybe_raise(False)

        # when `co_is_finished` the `exception_` is never None
        try:
            raise self.exception_

        finally:
            self.exception_ = StopIteration

    def resume(self, value: ... = None) -> None:
        """Resume them inside `.co_yield`"""
        if not self.co_is_suspended:
            raise self.OutOfSequenceError("`.resume` must be preceded by `.wait`")

        if not self.co_is_finished:
            self.exception_, self.value_ = None, value

        self.leave()

    def throw(self, exception: BaseException) -> None:
        """Raise an exception from `.co_yield`"""
        # XXX check exc type?
        if not self.co_is_suspended:
            raise self.OutOfSequenceError(".`throw` must be preceded by `.wait`")

        try:
            # loop back if they do not exist anymore
            if self.co_is_finished:
                raise exception

            self.exception_, self.value_ = exception, self.nothing

        finally:
            self.leave()

    def close(self):
        # nothing to close, if the coro hasn't started, or if it has already
        #  been closed
        if not self.is_alive():
            return

        try:
            # make sure we're inside the wait-resume section
            if not self.co_is_suspended:
                self.wait()

            # if the coro uses co-yield, then it knows its a generator, hence
            #  it respects the protocol
            if not self.co_is_finished:
                self.throw(GeneratorExit)
                self.wait()

            # here the coro is suspended and finished
        except (GeneratorExit, StopIteration):
            pass

        else:
            raise RuntimeError

        finally:
            self.join()

    def __iter__(self) -> ...:
        """A generator interface for the coroutine."""
        try:
            self.start()
            while True:
                try:
                    response = yield self.wait()

                except BaseException as e:
                    if self.exception_ is not None:
                        raise

                    self.throw(e)

                else:
                    self.resume(response)

        except StopIteration as e:
            return e.value

        finally:
            self.close()
