from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable


class BatchProcessor(Thread):  # SIMO
    """Collect batches of requests and process them with `target`"""

    timeout: float = 3.0

    def __init__(
        self,
        target: Callable,
        name: str = None,
        daemon: bool = None,
        *,
        maxsize: int = -1,
    ) -> None:
        super().__init__(name=name, daemon=daemon)
        self.exception_, self.target = None, target
        self.is_finished, self.requests = Event(), Queue(maxsize)

    def stop(self) -> None:
        self.is_finished.set()
        self.join()

    def run(self) -> None:
        batch = []
        while not self.is_finished.is_set():
            # collect the first element by a short-lived blocking call
            try:
                batch.append(self.requests.get(True, timeout=self.timeout))

            except Empty:
                continue

            # fetch all __immediately__ available items
            try:
                while True:
                    batch.append(self.requests.get_nowait())

            except Empty:
                pass

            # separate the data from the queues
            coms, inputs = zip(*batch)
            batch.clear()

            # process and send each result back to its origin, auto-shutdown in
            #  the case of emergency (`target` throws)
            try:
                for com, out in zip(coms, self.target(inputs)):
                    com.put_nowait(out)

            except Exception as e:
                self.exception_ = e
                break

        # make sure to inform all waiting `co_yield`-s about termination
        self.is_finished.set()

    def connect(self) -> Callable:
        """Establish a communications closure"""
        com = Queue()

        def co_yield(input: ...) -> ...:
            # send the request, unless the server has been terminated
            if self.is_finished.is_set():
                raise RuntimeError

            self.requests.put((com, input))

            # wait on the exclusive queue, making sure not to block for too long
            while not self.is_finished.is_set():
                try:
                    return com.get(True, timeout=self.timeout)

                except Empty:
                    continue

            raise self.exception_ or RuntimeError

        return co_yield
