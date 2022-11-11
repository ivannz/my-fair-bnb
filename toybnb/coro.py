from threading import Condition, Thread, Lock


class Coroutine(Thread):
    """My own coroutine for control inversion.

    Can use `return` and `raise` in the target, but have to call `.co_yield`
    instead of `yield`.
    """

    _nothing: object = object()  # a sentinel object

    def __init__(self, target, name=None, args=(), kwargs=None, daemon=None):
        super().__init__(None, target, name, args, kwargs, daemon=daemon)

        # cv is used for lock-step synchronisation, while its lock does the mutexing
        self.cv = Condition(Lock())
        self.is_running, self.co_is_finished = True, False
        self.exception_, self.value_ = self._nothing, self._nothing
        self.is_suspended = False

    def maybe_raise(self, co: bool) -> ...:
        """Either return the value of raise the pending exception."""
        try:
            assert not co or (not self.is_suspended and self.is_running)

            if self.exception_ is not self._nothing:
                assert co or self.co_is_finished
                raise self.exception_

            if not co and self.co_is_finished:
                raise StopIteration

            assert self.value_ is not self._nothing
            return self.value_

        finally:
            # reset the value and the exception
            self.exception_, self.value_ = self._nothing, self._nothing

    def co_acquire(self) -> None:
        """`.co_acquire` is followed by `.co_release` in our thread"""
        self.cv.acquire()
        self.cv.wait_for(lambda: self.is_running)

    def co_release(self) -> None:
        """our `.co_release` is coupled to their `.acquire`"""
        # assert not self.is_suspended

        self.is_running = False
        self.cv.notify(1)
        self.cv.release()

    def co_yield(self, value: ... = None) -> ...:
        """Unsuspend them at `.wait`, wait for their `.resume`"""
        # assert not self.is_suspended and self.is_running

        self.exception_, self.value_ = self._nothing, value
        self.co_release()  # set .is_running to False and notify waiting threads

        pass

        self.co_acquire()  # block until `.is_running == True`
        return self.maybe_raise(True)

    def co_raise(self, value: BaseException) -> None:
        """Raise an exception at `.wait`"""
        # assert not self.is_suspended and self.is_running
        self.co_is_finished = True
        # XXX if they raise at us, then they are sure that they cannot continue
        #  otherwise they wouldn't have raised at us in the first place.
        self.exception_, self.value_ = value, self._nothing
        self.co_release()

    def co_return(self, value: ... = None) -> None:
        # they return by raising StopIteration at out `.wait`
        self.co_raise(StopIteration(value))

    def run(self):
        self.is_running = True
        try:
            # Make sure the cv lock is acquired by them. Since they are
            #  running, the `.wait_for` immediately returns.
            self.co_acquire()
            self.co_return(self._target(*self._args, **self._kwargs))

        except BaseException as e:
            self.co_raise(e)

        finally:
            del self._target, self._args, self._kwargs

    def acquire(self) -> None:
        """`.acquire` is followed by `.release` in the same thread"""
        assert not self.is_suspended

        if not self.co_is_finished:
            self.cv.acquire()
            self.cv.wait_for(lambda: not self.is_running)

        self.is_suspended = True

    def release(self) -> None:
        """our `.release` is coupled to their `.co_acquire`"""
        if self.co_is_finished:
            return

        self.is_running = True
        self.cv.notify(1)
        self.is_suspended = False
        self.cv.release()

    def wait(self) -> ...:
        """Wait for their `.co_yield`"""
        self.acquire()
        return self.maybe_raise(False)

    def resume(self, value: ... = None) -> None:
        """Resume them inside `.co_yield`"""
        assert self.is_suspended

        self.exception_, self.value_ = self._nothing, value
        self.release()

    def throw(self, value: BaseException) -> None:
        """Raise an exception from `.co_yield`"""
        assert self.is_suspended

        # loop back if they do not exist anymore
        if self.co_is_finished:
            raise value

        self.exception_, self.value_ = value, self._nothing
        self.release()

    def close(self):
        if not self.is_alive():
            return

        try:
            if not self.is_suspended:
                self.wait()

            while not self.co_is_finished:
                self.throw(GeneratorExit)
                self.wait()

        except (GeneratorExit, StopIteration):
            pass

        else:
            raise RuntimeError

        finally:
            self.join()

        assert self.is_suspended

    def __iter__(self) -> ...:
        """A generator interface for the coroutine."""
        try:
            self.start()
            while True:
                request = self.wait()
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
