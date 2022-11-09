from threading import Condition, Thread


class Coroutine(Thread):
    """My own coroutine for control inversion."""

    _nothing: object = object()

    def __init__(self, target, name=None, *, daemon=None):
        super().__init__(target=target, name=name, daemon=daemon)

        self.cv = Condition()
        self.is_running, self.is_finished, self.is_closed = True, False, False
        self.exception_ = self._nothing
        self.request_, self.value_ = self._nothing, self._nothing

    def co_acquire(self) -> None:
        """`.co_acquire` is followed by `.co_release` in our thread"""
        self.cv.acquire()
        self.cv.wait_for(lambda: self.is_running)

    def co_release(self) -> None:
        """our `.co_release` is coupled to their `.acquire`"""
        self.is_running = False
        # notify anyone waiting then release our lock
        self.cv.notify(1)
        self.cv.release()

    def acquire(self) -> None:
        """`.acquire` is followed by `.release` in the same thread"""
        self.cv.acquire()
        self.cv.wait_for(lambda: not self.is_running)

    def release(self) -> None:
        """our `.release` is coupled to their `.co_acquire`"""
        self.is_running = True
        self.cv.notify(1)
        self.cv.release()

    def co_yield(self, value: ...) -> ...:
        self.exception_, self.value_ = self._nothing, value
        self.co_release()

        self.co_acquire()
        if self.exception_ is not self._nothing:
            raise self.exception_

        assert self.request_ is not self._nothing
        return self.request_

    def co_throw(self, value: BaseException) -> None:
        self.exception_, self.value_ = value, self._nothing
        self.co_release()

    def co_close(self):
        self.exception_, self.value_ = self._nothing, self._nothing
        self.is_finished = True
        self.co_release()

    def wait(self) -> ...:
        self.acquire()
        if self.exception_ is not self._nothing:
            raise self.exception_

        assert self.value_ is not self._nothing
        return self.value_

    def resume(self, value: ...) -> None:
        self.exception_, self.request_ = self._nothing, value
        self.release()

    def throw(self, value: BaseException) -> None:
        self.exception_, self.request_ = value, self._nothing
        self.release()

    def close(self):
        self.exception_, self.request_ = self._nothing, self._nothing
        self.is_closed = True
        self.release()

    def run(self):
        try:
            self.cv.acquire()
            self.is_running = True
            super().run()

        except BaseException as e:
            self.co_throw(e)

        finally:
            self.co_close()
