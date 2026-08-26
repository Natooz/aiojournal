"""Pool of function calls."""

from __future__ import annotations

import asyncio
import csv
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypeVar, cast

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

# Scalar and dictionary files deliberately place ``error`` differently. The header
# order determines the return format
RESULTS_FILE_COMMON_COLUMNS = (
    "task_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "success",
)
RESULTS_FILE_SCALAR_COLUMNS = (
    *RESULTS_FILE_COMMON_COLUMNS,
    "result",
    "error",
)
RESULTS_FILE_DICTIONARY_METADATA_COLUMNS = (*RESULTS_FILE_COMMON_COLUMNS, "error")
RESULTS_FILE_RESERVED_DICTIONARY_KEYS = frozenset(
    RESULTS_FILE_DICTIONARY_METADATA_COLUMNS
)


# TODO make possible to subclass it and add a run abstract method called instead of
#  provided func callable
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
        :raises ValueError: If ``max_concurrency`` is less than one or an existing
            results file has an incompatible header.
        """
        if max_concurrency < 1:
            raise ValueError(_ := "max_concurrency must be at least 1")

        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks = set()
        self.results_file_path = results_file_path
        self.csv_lock = asyncio.Lock()  # Lock to prevent concurrent writes to CSV
        self._returns_dictionary_results: bool | None = None
        self._dictionary_result_columns: tuple[str, ...] = ()

        # Restore the return format fixed by an existing results file. A file containing
        # only failed rows has not established its return format yet.
        if (
            results_file_path is not None
            and results_file_path.is_file()
            and results_file_path.stat().st_size > 0
        ):
            with results_file_path.open(newline="") as results_file:
                reader = csv.DictReader(results_file)
                if reader.fieldnames is not None:
                    # Scalar and dictionary files use different column ordering so a
                    # dictionary containing only "result" remains distinguishable.
                    if reader.fieldnames == list(RESULTS_FILE_SCALAR_COLUMNS):
                        # Adding success condition here as previous run might only have
                        # save failed tasks, which use the scalar format. In that case
                        # the returned format is still not determined.
                        if any(row["success"] == "True" for row in reader):
                            self._returns_dictionary_results = False
                    elif reader.fieldnames[
                        : len(RESULTS_FILE_DICTIONARY_METADATA_COLUMNS)
                    ] == list(RESULTS_FILE_DICTIONARY_METADATA_COLUMNS):
                        self._returns_dictionary_results = True
                        self._dictionary_result_columns = tuple(
                            reader.fieldnames[
                                len(RESULTS_FILE_DICTIONARY_METADATA_COLUMNS) :
                            ]
                        )
                    else:
                        message = "Existing results file has an incompatible CSV header"
                        raise ValueError(message)

    def _create_csv_headers(self) -> None:
        """Create the CSV file with headers."""
        if self.results_file_path is None:
            return
        if not self.results_file_path.parent.is_dir():
            self.results_file_path.parent.mkdir(parents=True)
        with self.results_file_path.open("w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    *RESULTS_FILE_DICTIONARY_METADATA_COLUMNS,
                    *self._dictionary_result_columns,
                ]
                if self._returns_dictionary_results
                else RESULTS_FILE_SCALAR_COLUMNS
            )

    def _expand_results_file_headers(self) -> None:
        """Add inferred dictionary columns while preserving previous failed rows."""
        if self.results_file_path is None:
            return
        temporary_results_file_path = self.results_file_path.with_suffix(
            f"{self.results_file_path.suffix}.tmp"
        )

        with self.results_file_path.open(newline="") as results_file:
            reader = csv.DictReader(results_file)
            with temporary_results_file_path.open("w", newline="") as temporary_file:
                writer = csv.DictWriter(
                    temporary_file,
                    fieldnames=[
                        *RESULTS_FILE_DICTIONARY_METADATA_COLUMNS,
                        *self._dictionary_result_columns,
                    ],
                )
                writer.writeheader()

                # Existing rows are failures written before the return format was known.
                # Copy their shared metadata and leave inferred result columns empty.
                for row in reader:
                    writer.writerow(
                        {
                            column_name: row.get(column_name, "")
                            for column_name in writer.fieldnames
                        }
                    )

        temporary_results_file_path.replace(self.results_file_path)

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
        :param result: Scalar task result or dictionary of result-column values.
        :param error: Error message when the task failed.
        :raises TypeError: If a successful result does not match the established return
            format or contains a non-string dictionary key.
        :raises ValueError: If a dictionary cannot establish or does not match the
            inferred column schema.
        """
        if self.results_file_path is None:
            return

        # Format timestamps
        start_time_str = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()
        end_time_str = datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat()
        duration = end_time - start_time

        # Ensure only one task writes to the CSV at a time
        async with self.csv_lock:
            result_is_dictionary = isinstance(result, dict)

            # Check every successful result. The first fixes the scalar or dictionary
            # format; later results must keep that type and cannot add columns.
            if success and result_is_dictionary:
                # Validate column names
                if not all(isinstance(column_name, str) for column_name in result):
                    message = "Dictionary results must only use string keys"
                    raise TypeError(message)
                dictionary_result = cast("dict[str, Any]", result)
                reserved_column_names_in_result = (
                    set(dictionary_result) & RESULTS_FILE_RESERVED_DICTIONARY_KEYS
                )
                if reserved_column_names_in_result:
                    names = ", ".join(sorted(reserved_column_names_in_result))
                    message = f"Dictionary result uses reserved columns: {names}"
                    raise ValueError(message)

                # Infer every result column from the first successful dictionary.
                returned_dictionary_result_columns = tuple(dictionary_result)
                if self._returns_dictionary_results is None:
                    # This is the first successful return, so its key order permanently
                    # defines the dictionary columns for this results file.
                    self._returns_dictionary_results = True
                    self._dictionary_result_columns = returned_dictionary_result_columns
                    # If result file already existed, because of previous failed tasks,
                    # rewrite it with the now determined columns names
                    if (
                        self.results_file_path.is_file()
                        and self.results_file_path.stat().st_size > 0
                    ):
                        self._expand_results_file_headers()
                elif not self._returns_dictionary_results:
                    message = "Expected a scalar result, received a dictionary"
                    raise TypeError(message)

                # Once inferred, later dictionaries may omit columns but cannot add any.
                unexpected_result_columns = set(
                    returned_dictionary_result_columns
                ) - set(self._dictionary_result_columns)
                if unexpected_result_columns:
                    names = ", ".join(sorted(unexpected_result_columns))
                    message = f"Dictionary result contains unexpected columns: {names}"
                    raise ValueError(message)

            # Scalar result format
            elif success:
                if self._returns_dictionary_results:
                    message = "Expected a dictionary result, received a scalar"
                    raise TypeError(message)
                self._returns_dictionary_results = False

            # Create the file after a successful result has had a chance to infer its
            # columns. Failed tasks create the unresolved scalar-style header.
            if (
                not self.results_file_path.is_file()
                or self.results_file_path.stat().st_size == 0
            ):
                self._create_csv_headers()

            # Project dictionary values onto the inferred columns. Missing keys and
            # failed-task dictionary values become empty CSV cells.
            dictionary_result_values = (
                [
                    result.get(column_name, "")
                    for column_name in self._dictionary_result_columns
                ]
                if result_is_dictionary
                else [""] * len(self._dictionary_result_columns)
            )

            # Append one complete task record using the established column order.
            with self.results_file_path.open("a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                task_metadata = [
                    task_id,
                    start_time_str,
                    end_time_str,
                    f"{duration:.4f}",
                    success,
                ]
                if self._returns_dictionary_results:
                    # Dictionary files store error metadata first, followed by values in
                    # the exact order established by the first dictionary result.
                    writer.writerow(
                        [*task_metadata, error or "", *dictionary_result_values]
                    )
                else:
                    # Scalar and unresolved failure rows keep the standard result column
                    # immediately before the error column.
                    writer.writerow(
                        [
                            *task_metadata,
                            str(result) if result is not None else "",
                            error or "",
                        ]
                    )

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        task_id: str,
        *,
        _first_task_exception_future: asyncio.Future[Exception] | None = None,
        **kwargs,
    ) -> T:
        """
        Execute an async callable within the configured concurrency limit.

        When a results file is configured, the attempt is appended to its CSV journal.

        :param func: Async callable to execute.
        :param task_id: Identifier written to the CSV journal.
        :param _first_task_exception_future: Internal batch signal containing the first
            task exception.
        :param kwargs: Keyword arguments passed to ``func``.
        :return: The callable's result.
        :raises Exception: Any exception raised by ``func``.
        """
        # Start timing only after a concurrency slot is available.
        async with self.semaphore:
            if (
                _first_task_exception_future is not None
                and _first_task_exception_future.done()
            ):
                raise asyncio.CancelledError

            error_msg = None
            result = None
            success = False
            start_time = time.time()

            try:
                result = await func(**kwargs)
                success = True
                return result

            except Exception as exception:
                if (
                    _first_task_exception_future is not None
                    and not _first_task_exception_future.done()
                ):
                    _first_task_exception_future.set_result(exception)
                error_msg = str(exception)
                logger.info(
                    _ := f"{task_id} - Task failed: {error_msg}",
                    extra={"task_id": task_id, "error": error_msg},
                )
                raise

            finally:
                # Save result to CSV
                if self.results_file_path is not None:
                    # Saving can fail if the result has the wrong type or dictionary
                    # keys, or if the file cannot be written. Signal that failure before
                    # releasing the semaphore so queued calls do not start.
                    try:
                        await self._save_to_csv(
                            task_id=task_id,
                            start_time=start_time,
                            end_time=time.time(),
                            success=success,
                            result=result,
                            error=error_msg,
                        )
                    except Exception as exception:
                        if (
                            _first_task_exception_future is not None
                            and not _first_task_exception_future.done()
                        ):
                            _first_task_exception_future.set_result(exception)
                        raise

    async def gather(
        self,
        funcs_with_args: list[dict[str, Any]],
        task_ids: list[str],
        progress_bar_desc: str | None = None,
        *,
        raise_on_failure: bool = False,
    ) -> list[Any]:
        """
        Execute multiple functions concurrently.

        :param funcs_with_args: Call specifications containing a ``func`` entry and
            keyword arguments for that callable.
        :param task_ids: Task identifiers corresponding to ``funcs_with_args``.
        :param progress_bar_desc: Progress-bar description, or ``None`` to disable the
            progress bar.
        :param raise_on_failure: Stop queued calls after the first task exception, wait
            for active calls to finish, and then raise that exception.
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

        first_task_exception_future: asyncio.Future[Exception] | None = (
            asyncio.get_running_loop().create_future() if raise_on_failure else None
        )

        # Create tasks and start them immediately
        tasks = []
        for item, task_id in zip(funcs_with_args, task_ids, strict=True):
            item_copy = item.copy()
            func = item_copy.pop("func")
            # Create a task that will call the function with semaphore management
            task = asyncio.create_task(
                self.call(
                    func,
                    task_id=task_id,
                    _first_task_exception_future=first_task_exception_future,
                    **item_copy,
                )
            )
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

        if first_task_exception_future is not None:
            if first_task_exception_future.done():
                raise first_task_exception_future.result()
            for task in tasks:
                if task_exception := task.exception():
                    raise task_exception

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
            except (Exception, asyncio.CancelledError) as exception:  # noqa: BLE001
                return exception

        return await tqdm_asyncio.gather(*map(wrap, fs), **kwargs)

    async def map(
        self,
        func: Callable[..., Awaitable[T]],
        args_list: list[dict[str, Any]],
        task_ids: list[str],
        progress_bar_desc: str | None = None,
        *,
        raise_on_failure: bool = False,
    ) -> list[T | BaseException]:
        """
        Apply one async callable to multiple argument sets concurrently.

        :param func: Async callable to execute.
        :param args_list: Keyword-argument dictionaries passed to ``func``.
        :param task_ids: Task identifiers corresponding to ``args_list``.
        :param progress_bar_desc: Progress-bar description, or ``None`` to disable the
            progress bar.
        :param raise_on_failure: Stop queued calls after the first task exception, wait
            for active calls to finish, and then raise that exception.
        :return: Results or exceptions in the same order as ``args_list``.
        :raises ValueError: If the number of task IDs does not match the number of
            argument sets.
        """
        funcs_with_args = [{"func": func, **args} for args in args_list]
        return await self.gather(
            funcs_with_args,
            task_ids,
            progress_bar_desc,
            raise_on_failure=raise_on_failure,
        )

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
            if (fieldnames := reader.fieldnames) is None:
                return
            with temp_path.open("w") as tmp_file:
                writer = csv.DictWriter(tmp_file, fieldnames=fieldnames)
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
