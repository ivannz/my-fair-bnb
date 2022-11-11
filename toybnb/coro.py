from threading import Condition, Thread, Lock
from time import sleep


class Coroutine(Thread):
    """My own coroutine for control inversion.

    Can use `return` and `raise` in the target, but have to call `.co_yield`
    instead of `yield`.
    """

    nothing: object = object()  # a sentinel object

    def __init__(self, target, name=None, args=(), kwargs=None, daemon=None):
        super().__init__(None, target, name, args, kwargs, daemon=daemon)

        # cv is used for lock-step synchronisation, while its lock is the mutex
        # """All methods of `Lock` are executed atomically."""
        # https://docs.python.org/3/library/threading.html#lock-objects
        self.cv = Condition(Lock())
        self.exception_, self.value_ = self.nothing, self.nothing
        self.co_is_running, self.co_is_suspended = False, False
        self.co_is_finished = False

    def maybe_raise(self, co: bool) -> ...:
        """Either return the value of raise the pending exception."""
        try:
            if self.exception_ is not self.nothing:
                assert co or self.co_is_finished
                raise self.exception_

            if not co and self.co_is_finished:
                raise StopIteration

            assert self.value_ is not self.nothing
            return self.value_

        finally:
            # reset the value and the exception
            # self.exception_, self.value_ = self.nothing, self.nothing
            self.value_ = self.nothing
            if co:
                # we threw at them, so when they handle the exception it must
                #  be cleared. If the exception originated from them, then
                #  we do not clear it. On our side it would also be clear
                #  if the exception was an interrupt
                self.exception_ = self.nothing

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
        self.exception_, self.value_ = self.nothing, value
        self.co_leave()

        if stall > 0:
            sleep(stall)

        self.co_enter()
        return self.maybe_raise(True)

    def co_raise(self, value: BaseException) -> None:
        """Raise an exception at `.wait`"""
        self.co_is_finished = True
        # XXX if they raise at us, then they are sure that they cannot continue
        #  otherwise they wouldn't have raised at us in the first place.
        self.exception_, self.value_ = value, self.nothing
        self.co_leave()

    def co_return(self, value: ... = None) -> None:
        # they return by raising StopIteration at our `.wait`
        exception = StopIteration if value is None else StopIteration(value)
        return self.co_raise(exception)

    def run(self):
        self.co_is_running = True
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
        assert not self.co_is_suspended

        self.co_is_suspended = self.cv.acquire()  # atomic, but can be interrupted
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
        # if not self._started.is_set():
        #     raise RuntimeError

        # If we own the time-slice, then we know that they are inside `co_yield
        #  and are blocked. if they own, then we block only if we need the
        #  time-slice.
        self.enter()
        return self.maybe_raise(False)

    def resume(self, value: ... = None) -> None:
        """Resume them inside `.co_yield`"""
        self.exception_, self.value_ = self.nothing, value
        self.leave()

    def throw(self, value: BaseException) -> None:
        """Raise an exception from `.co_yield`"""
        # loop back if they do not exist anymore
        if self.co_is_finished:
            raise value

        self.exception_, self.value_ = value, self.nothing
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
                    request = self.wait()

                except BaseException as e:
                    if self.exception_ is not self.nothing:
                        raise

                    self.throw(e)
                    continue

                try:
                    response = yield request

                except BaseException as e:
                    self.throw(e)

                else:
                    self.resume(response)

        except StopIteration as e:
            return e.value

        finally:
            self.close()
