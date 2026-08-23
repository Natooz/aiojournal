"""Tests for the asynchronous batch runner."""

import asyncio
from pathlib import Path

import pytest

from aiojournal import AsyncBatchRunner


async def test_map_preserves_order_and_limits_concurrency() -> None:
    num_active_calls = 0
    max_num_active_calls = 0

    async def tracked_double(value: int) -> int:
        nonlocal max_num_active_calls, num_active_calls
        num_active_calls += 1
        max_num_active_calls = max(max_num_active_calls, num_active_calls)
        await asyncio.sleep(0.01)
        num_active_calls -= 1
        return value * 2

    runner = AsyncBatchRunner(max_concurrency=2)
    results = await runner.map(
        tracked_double,
        args_list=[{"value": value} for value in range(4)],
        task_ids=[str(value) for value in range(4)],
    )

    assert results == [0, 2, 4, 6]
    assert max_num_active_calls == 2


async def test_gather_returns_exceptions_and_records_results(tmp_path: Path) -> None:
    failure_message = "failed on purpose"

    async def succeed(value: int) -> int:
        return value

    async def fail() -> None:
        raise ValueError(failure_message)

    runner = AsyncBatchRunner(results_file_path=tmp_path / "nested" / "results.csv")
    results = await runner.gather(
        [{"func": succeed, "value": 3}, {"func": fail}],
        task_ids=["success", "failure"],
    )
    recorded_results_by_task_id = {
        result["task_id"]: result for result in runner.read_results_csv()
    }

    assert results[0] == 3
    assert isinstance(results[1], ValueError)
    assert recorded_results_by_task_id["success"]["success"] == "True"
    assert recorded_results_by_task_id["success"]["result"] == "3"
    assert recorded_results_by_task_id["failure"]["success"] == "False"
    assert recorded_results_by_task_id["failure"]["error"] == failure_message


async def test_failed_attempt_cleanup_keeps_successes_and_unresolved_failures(
    tmp_path: Path,
) -> None:
    failure_message = "temporary failure"

    async def fail() -> None:
        raise RuntimeError(failure_message)

    async def succeed() -> str:
        return "done"

    runner = AsyncBatchRunner(results_file_path=tmp_path / "results.csv")
    await runner.gather(
        [{"func": fail}, {"func": succeed}, {"func": fail}],
        task_ids=["retried", "retried", "unresolved"],
    )

    runner.remove_tasks_failed_from_res_file()
    remaining_results_by_task_id = {
        result["task_id"]: result for result in runner.read_results_csv()
    }

    assert set(remaining_results_by_task_id) == {"retried", "unresolved"}
    assert remaining_results_by_task_id["retried"]["success"] == "True"
    assert remaining_results_by_task_id["unresolved"]["success"] == "False"
    assert remaining_results_by_task_id["unresolved"]["error"] == failure_message


async def test_rejects_invalid_batch_configuration() -> None:
    with pytest.raises(ValueError, match="max_concurrency must be at least 1"):
        AsyncBatchRunner(max_concurrency=0)

    runner = AsyncBatchRunner()
    assert list(runner.read_results_csv()) == []

    with pytest.raises(ValueError, match="Length of task_ids"):
        await runner.gather([{"func": asyncio.sleep}], task_ids=[])
