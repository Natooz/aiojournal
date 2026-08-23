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


class AsyncBatchRunner:
    """A simple async runner that limits the number of concurrent function calls."""

    def __init__(
        self, max_concurrency: int = MAX_CONCURRENCY, csv_path: Path | None = None
    ) -> None:
        """
        Initialize the async pool with a maximum concurrency limit.

        Args:
        ----
            max_concurrency: Maximum number of concurrent function calls allowed
            csv_path: Path to the CSV file where results will be saved

        """
        if max_concurrency < 1:
            raise ValueError(_ := "max_concurrency must be at least 1")

        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks = set()
        self.csv_path = csv_path
        self.csv_lock = asyncio.Lock()  # Lock to prevent concurrent writes to CSV

    def _create_csv_headers(self) -> None:
        """Create the CSV file with headers."""
        if not self.csv_path.parent.is_dir():
            self.csv_path.parent.mkdir(parents=True)
        with self.csv_path.open("w") as csvfile:
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
        Save a function call result to the CSV file.

        Args:
        ----
            task_id: Unique identifier for the task
            start_time: Timestamp when the task started
            end_time: Timestamp when the task ended
            success: Whether the task completed successfully
            result: Result of the function call (will be converted to string)
            error: Error message if the task failed

        """
        # Format timestamps
        start_time_str = datetime.fromtimestamp(start_time, tz=UTC).isoformat()
        end_time_str = datetime.fromtimestamp(end_time, tz=UTC).isoformat()
        duration = end_time - start_time

        # Ensure only one task writes to the CSV at a time
        async with self.csv_lock:
            if not self.csv_path.is_file():
                self._create_csv_headers()
            with self.csv_path.open("a") as csvfile:
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
        Execute a function with the given arguments, respecting the concurrency limit.

        Save the result to CSV file.

        Args:
        ----
            func: Async callable to execute
            task_id: Optional identifier for the task (auto-generated if None)
            **kwargs: Arguments to pass to the function

        Returns:
        -------
            The result of the function call

        """
        start_time = None
        error_msg = None
        result = None
        success = False

        try:
            # Acquire semaphore before executing
            async with self.semaphore:
                start_time = time.time()
                result = await func(**kwargs)
                success = True
                return result

        except Exception as e:
            error_msg = str(e)
            logger.info(
                _ := f"{task_id} - Task failed: {error_msg}",
                extra={"task_id": task_id, "error": error_msg},
            )
            raise

        finally:
            end_time = time.time()
            # Save result to CSV
            if self.csv_path is not None:
                await self._save_to_csv(
                    task_id=task_id,
                    start_time=start_time,
                    end_time=end_time,
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
        Execute multiple functions with their arguments concurrently.

        Args:
        ----
            funcs_with_args: List of dictionaries, each containing:
                - 'func': the async callable
                - any other key-value pairs are passed as arguments to the function
            task_ids: List of task IDs
            progress_bar_desc: Description for the progress bar. Leave ``None`` for no
                progress bar. (default: None)

        Returns:
        -------
            List of results in the same order as the input functions

        """
        if not funcs_with_args:
            return []

        if len(task_ids) != len(funcs_with_args):
            raise ValueError(
                _ := "Length of task_ids must match length of funcs_with_args"
            )

        # Create tasks and start them immediately
        tasks = []
        for item, task_id in zip(funcs_with_args, task_ids, strict=False):
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
        """Wrap ``tqdm_asyncio.gather`` to supports ``return_exceptions``."""
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
    ) -> list[T]:
        """
        Apply the same function to multiple sets of arguments concurrently.

        Args:
            func: Async callable to execute
            args_list: List of argument dictionaries to pass to the function
            task_ids: Optional list of task IDs (auto-generated if None)
            progress_bar_desc: Description for the progress bar. Leave ``None`` for no
                progress bar. (default: None)

        Returns:
            List of results in the same order as the input arguments

        """
        funcs_with_args = [{"func": func, **args} for args in args_list]
        return await self.gather(funcs_with_args, task_ids, progress_bar_desc)

    def get_active_count(self) -> int:
        """Get the number of currently running tasks."""
        return self.max_concurrency - self.semaphore._value

    def read_results_csv(
        self, csv_path: str | None = None, as_dict: bool = False
    ) -> Generator[dict[str, str]] | Generator[tuple[str, dict[str, str]]]:
        """
        Read the results CSV file, yielding one row at a time.

        :param csv_path: Path to the CSV file
        :param as_dict: if ``True``, yield ``(task_id, row)`` tuples instead of row
            dicts. (default: ``False``)
        :return: Generator yielding rows from the CSV
        """
        if csv_path is None:
            csv_path = self.csv_path

        if not csv_path.is_file():
            return

        with csv_path.open() as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if not as_dict:
                    yield row
                else:
                    row_id = row.pop("task_id")
                    yield row_id, row

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
            (default: ``True``)
        """
        if self.csv_path is None or not self.csv_path.is_file():
            return

        csv_path = self.csv_path
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
