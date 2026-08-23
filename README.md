# aiojournal

An asynchronous batch execution runner with results persistence and resumability.

Especially useful when making concurrent API calls and directly save each request's result.
For example when benchmarking LLM models, from big providers or inference engines.

```bash
pip install aiojournal
```

## Example

```python
import asyncio
from pathlib import Path

from aiojournal import AsyncBatchRunner


async def double(value: int) -> int:
    await asyncio.sleep(0.1)
    return value * 2


async def main() -> None:
    runner = AsyncBatchRunner(
        max_concurrency=2,
        results_file_path=Path("tasks.csv"),
    )
    results = await runner.map(
        double,
        args_list=[{"value": 1}, {"value": 2}, {"value": 3}],
        task_ids=["one", "two", "three"],
    )
    print(results)  # [2, 4, 6]


asyncio.run(main())
```

Results remain in input order. Each task appends a row containing its ID, timestamps,
duration, result, and error status to `tasks.csv`. If a task fails, `map` and
`gather` place the exception in that task's result position so the rest of the batch can
finish.

The journal can be read without loading the whole file:

```python
for result in runner.read_results_csv():
    print(result["task_id"], result["success"], result["result"])
```
