"""Pool of function calls."""

from __future__ import annotations

import asyncio
import csv
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

from tqdm.asyncio import tqdm_asyncio

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator
    from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar("T")  # Return type for the callable

MAX_CONCURRENCY = 100


# TODO make possible to subclass it and add a run abstract method called instead of
#  provided func callable
# TODO save custom rows in csv (returned dictionary from executed function)
# TODO raise on failure option
class AsyncBatchRunner:
    """A simple async runner that limits the number of concurrent function calls."""

    def __init__(
        self,
        max_concurrency: int = MAX_CONCURRENCY,
        results_file_path: Path | None = None,
    ) -> None:
        """
        Initialize the runner and its concurrency limit.

        :param max_concurrency: Maximum number of concurrent function calls.
        :param results_file_path: File where task results are appended, or ``None`` to
            disable journaling.
        :raises ValueError: If ``max_concurrency`` is less than one.
        """
        if max_concurrency < 1:
            raise ValueError(_ := "max_concurrency must be at least 1")

        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks = set()
        self.results_file_path = results_file_path
        self.csv_lock = asyncio.Lock()  # Lock to prevent concurrent writes to CSV

    def _create_csv_headers(self) -> None:
        """Create the CSV file with headers."""
        if not self.results_file_path.parent.is_dir():
            self.results_file_path.parent.mkdir(parents=True)
        with self.results_file_path.open("w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "task_id",
                    "start_time",
                    "end_time",
                    "duration_seconds",
                    "success",
                    "result",
                    "error",
                ]
            )

    async def _save_to_csv(
        self,
        task_id: str,
        start_time: float,
        end_time: float,
        success: bool,
        result: Any,  # noqa:ANN401
        error: str | None = None,
    ) -> None:
        """
        Append a function-call result to the CSV journal.

        :param task_id: Identifier of the task.
        :param start_time: Unix timestamp recorded when the task started.
        :param end_time: Unix timestamp recorded when the task ended.
        :param success: Whether the task completed successfully.
        :param result: Task result, converted to a string before writing.
        :param error: Error message when the task failed.
        """
        # Format timestamps
        start_time_str = datetime.fromtimestamp(start_time, tz=UTC).isoformat()
        end_time_str = datetime.fromtimestamp(end_time, tz=UTC).isoformat()
        duration = end_time - start_time

        # Ensure only one task writes to the CSV at a time
        async with self.csv_lock:
            if not self.results_file_path.is_file():
                self._create_csv_headers()
            with self.results_file_path.open("a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        task_id,
                        start_time_str,
                        end_time_str,
                        f"{duration:.4f}",
                        success,
                        str(result) if result is not None else "",
                        error or "",
                    ]
                )

    async def call(
        self, func: Callable[..., Awaitable[T]], task_id: str | None = None, **kwargs
    ) -> T:
        """
        Execute an async callable within the configured concurrency limit.

        When a results file is configured, the attempt is appended to its CSV journal.

        :param func: Async callable to execute.
        :param task_id: Identifier written to the CSV journal.
        :param kwargs: Keyword arguments passed to ``func``.
        :return: The callable's result.
        :raises Exception: Any exception raised by ``func``.
        """
        # Start timing only after a concurrency slot is available.
        async with self.semaphore:
            error_msg = None
            result = None
            success = False

            try:
                start_time = time.time()
                result = await func(**kwargs)
                success = True
                return result

            except Exception as exception:
                error_msg = str(exception)
                logger.info(
                    _ := f"{task_id} - Task failed: {error_msg}",
                    extra={"task_id": task_id, "error": error_msg},
                )
                raise

            finally:
                # Save result to CSV
                if self.results_file_path is not None:
                    await self._save_to_csv(
                        task_id=task_id,
                        start_time=start_time,
                        end_time=time.time(),
                        success=success,
                        result=result,
                        error=error_msg,
                    )

    async def gather(
        self,
        funcs_with_args: list[dict[str, Any]],
        task_ids: list[str],
        progress_bar_desc: str | None = None,
    ) -> list[Any]:
        """
        Execute multiple functions concurrently.

        :param funcs_with_args: Call specifications containing a ``func`` entry and
            keyword arguments for that callable.
        :param task_ids: Task identifiers corresponding to ``funcs_with_args``.
        :param progress_bar_desc: Progress-bar description, or ``None`` to disable the
            progress bar.
        :return: Results or exceptions in the same order as ``funcs_with_args``.
        :raises ValueError: If the number of task IDs does not match the number of call
            specifications.
        """
        if not funcs_with_args:
            return []

        if len(task_ids) != len(funcs_with_args):
            raise ValueError(
                _ := "Length of task_ids must match length of funcs_with_args"
            )

        # Create tasks and start them immediately
        tasks = []
        for item, task_id in zip(funcs_with_args, task_ids, strict=True):
            item_copy = item.copy()
            func = item_copy.pop("func")
            # Create a task that will call the function with semaphore management
            task = asyncio.create_task(self.call(func, task_id=task_id, **item_copy))
            tasks.append(task)

        # Execute with or without progress bar
        if progress_bar_desc is not None:
            results = await self.tqdm_gather(
                *tasks, desc=progress_bar_desc, return_exceptions=True
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Remove completed tasks
        for task in tasks:
            self._tasks.discard(task)

        return results

    @staticmethod
    async def tqdm_gather(
        *fs: Awaitable[T],
        return_exceptions: bool = False,
        **kwargs,
    ) -> list[T | BaseException]:
        """
        Gather awaitables with progress and optional exception results.

        :param fs: Awaitables to execute.
        :param return_exceptions: Return exceptions alongside successful results when
            ``True``.
        :param kwargs: Keyword arguments passed to ``tqdm_asyncio.gather``.
        :return: Results in the same order as ``fs``.
        """
        if not return_exceptions:
            return await tqdm_asyncio.gather(*fs, **kwargs)

        async def wrap(f: Awaitable[T]) -> T | BaseException:
            try:
                return await f
            except Exception as e:  # noqa:BLE001
                return e

        return await tqdm_asyncio.gather(*map(wrap, fs), **kwargs)

    async def map(
        self,
        func: Callable[..., Awaitable[T]],
        args_list: list[dict[str, Any]],
        task_ids: list[str],
        progress_bar_desc: str | None = None,
    ) -> list[T | BaseException]:
        """
        Apply one async callable to multiple argument sets concurrently.

        :param func: Async callable to execute.
        :param args_list: Keyword-argument dictionaries passed to ``func``.
        :param task_ids: Task identifiers corresponding to ``args_list``.
        :param progress_bar_desc: Progress-bar description, or ``None`` to disable the
            progress bar.
        :return: Results or exceptions in the same order as ``args_list``.
        :raises ValueError: If the number of task IDs does not match the number of
            argument sets.
        """
        funcs_with_args = [{"func": func, **args} for args in args_list]
        return await self.gather(funcs_with_args, task_ids, progress_bar_desc)

    @property
    def num_active_count(self) -> int:
        """
        Return the number of currently active calls.

        :return: Number of acquired concurrency slots.
        """
        return self.max_concurrency - self.semaphore._value

    def read_results_csv(
        self, csv_path: Path | None = None
    ) -> Generator[dict[str, str]]:
        """
        Read the results CSV file, yielding one row at a time.

        :param csv_path: CSV file to read, or ``None`` to use the configured journal.
        :return: Rows from the CSV journal.
        """
        if csv_path is None:
            csv_path = self.results_file_path

        if csv_path is None or not csv_path.is_file():
            return

        with csv_path.open() as csvfile:
            reader = csv.DictReader(csvfile)
            yield from reader

    def remove_tasks_failed_from_res_file(
        self, remove_only_those_succeeded: bool = True
    ) -> None:
        """
        Remove rows marked as failed from the pool's results file.

        This streams the CSV to a temporary file and replaces the original to avoid
        loading the entire file in memory.

        :param remove_only_those_succeeded: If ``True``, only remove rows that failed
            for which another row with the same task_id is succeeded. This allows to
            clean multi-attempt failures while keeping failures with no success.
        """
        if self.results_file_path is None or not self.results_file_path.is_file():
            return

        csv_path = self.results_file_path
        temp_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")

        # First get ids of requests succeeded
        seen_success_ids: set[str] = set()
        if remove_only_those_succeeded:
            with csv_path.open() as csvfile:
                reader = csv.DictReader(csvfile)
                if reader.fieldnames is None:
                    return
                for row in reader:
                    if row["success"] == "True":
                        seen_success_ids.add(row["task_id"])

        # Write rows with success or failed without success
        with csv_path.open() as csvfile:
            reader = csv.DictReader(csvfile)
            with temp_path.open("w") as tmp_file:
                writer = csv.DictWriter(tmp_file, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    # Write success rows and failed one with no success (same task_id)
                    # if requested
                    if row["success"] == "True" or (
                        remove_only_those_succeeded
                        and row["task_id"] not in seen_success_ids
                    ):
                        writer.writerow(row)

        temp_path.replace(csv_path)
