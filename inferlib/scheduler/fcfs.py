import asyncio

from asyncio import Future
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class QueueItem(Generic[T, R]):
    payload: T
    fut: Future[R]


class Scheduler(Generic[T, R]):
    def __init__(
        self,
        fn_to_call: Callable,
        batch_size: int = 4,
        queue_size: int = 100,
    ):
        self._queue: asyncio.Queue[QueueItem[T, R]] = asyncio.Queue(maxsize=queue_size)
        self._fn = fn_to_call
        self._batch_size = batch_size
        self._task = None
        self._lock = asyncio.Lock()

    async def start(self):
        async with self._lock:
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        async with self._lock:
            if self._task is None:
                return
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

            self._task = None

            while not self._queue.empty():
                item = self._queue.get_nowait()
                if not item.fut.done():
                    item.fut.set_exception(asyncio.CancelledError("Scheduler stopped"))

                self._queue.task_done()

    async def submit(self, payload: T):
        loop = asyncio.get_running_loop()
        fut: Future[R] = loop.create_future()
        # TODO: input processor to process
        queue_item = QueueItem(payload=payload, fut=fut)
        await self._queue.put(queue_item)
        return await fut

    async def _worker_loop(self):
        while True:
            batch: list[QueueItem] = [
                await self._queue.get() for _ in range(self._batch_size)
            ]
            payloads = [item.payload for item in batch]
            try:
                results = await asyncio.to_thread(self._fn, payloads)
                # TODO: output processor to process the results
                if len(batch) != len(results):
                    raise RuntimeError()

            except Exception as e:
                for item in batch:
                    if not item.fut.done():
                        item.fut.set_exception(e)
            else:
                for item, result in zip(batch, results):
                    if not item.fut.done():
                        item.fut.set_result(result)
            finally:
                for _ in batch:
                    self._queue.task_done()
