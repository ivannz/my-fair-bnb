from threading import Condition, Thread


class Coroutine(Thread):
    """My own coroutine for control inversion.

    Can use `return` and `raise` in the target, but have to call `.co_yield`
    instead of `yield`.
    """

    _nothing: object = object()  # a sentinel object

    def __init__(self, target, name=None, args=(), kwargs=None, *, daemon=None):
        super().__init__(None, target, name, args, kwargs, daemon=daemon)

        self.cv = Condition()
        self.run, self.is_running = self.co_run, True
        self.exception_, self.value_ = self._nothing, self._nothing

    def co_acquire(self) -> None:
        """`.co_acquire` is followed by `.co_release` in our thread"""
        self.cv.acquire()
        self.cv.wait_for(lambda: self.is_running)

    def co_release(self) -> None:
        """our `.co_release` is coupled to their `.acquire`"""
        self.is_running = False
        self.cv.notify(1)
        self.cv.release()

    def co_yield(self, value: ...) -> ...:
        """Unsuspend them at `.wait`, wait for their `.resume`"""
        self.exception_, self.value_ = self._nothing, value
        self.co_release()

        pass

        self.co_acquire()
        if self.exception_ is not self._nothing:
            raise self.exception_

        if self.value_ is not self._nothing:
            return self.value_

        raise RuntimeError

    def co_raise(self, value: BaseException) -> None:
        """Raise an exception at `.wait`"""
        self.exception_, self.value_ = value, self._nothing
        self.co_release()

    def co_return(self, value: ...) -> None:
        self.co_raise(StopIteration(value))

    def co_run(self):
        self.is_running = True
        try:
            self.co_acquire()
            self.co_return(self._target(*self._args, **self._kwargs))

        except BaseException as e:
            self.co_raise(e)

        finally:
            del self._target, self._args, self._kwargs

    def acquire(self) -> None:
        """`.acquire` is followed by `.release` in the same thread"""
        self.cv.acquire()
        self.cv.wait_for(lambda: not self.is_running)

    def release(self) -> None:
        """our `.release` is coupled to their `.co_acquire`"""
        self.is_running = True
        self.cv.notify(1)
        self.cv.release()

    def wait(self) -> ...:
        """Wait for their `.co_yield`"""
        self.acquire()
        if self.exception_ is not self._nothing:
            raise self.exception_

        if self.value_ is not self._nothing:
            return self.value_

        raise RuntimeError

    def resume(self, value: ...) -> None:
        """Resume them inside `.co_yield`"""
        self.exception_, self.value_ = self._nothing, value
        self.release()

    def throw(self, value: BaseException) -> None:
        """Raise an exception from `.co_yield`"""
        self.exception_, self.value_ = value, self._nothing
        self.release()

    def close(self):
        try:
            self.throw(GeneratorExit)
            self.wait()

        except (GeneratorExit, StopIteration):
            pass

        else:
            raise RuntimeError
