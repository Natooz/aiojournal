# aiojournal

An asynchronous batch execution runner with results persistence and resumability.

Especially useful when making concurrent API calls while directly saving each request's result, for example when benchmarking LLM models, from big providers or inference engines.
It is minimal and lightweight, only dependency is [tqdm](https://github.com/tqdm/tqdm).

```bash
pip install aiojournal
```

## Example

```python
import asyncio
from pathlib import Path

from aiojournal import AsyncBatchRunner


async def describe(value: int) -> dict[str, int | str]:
    await asyncio.sleep(0.1)
    return {
        "result": value * 2,
        "input": value,
        "parity": "even" if value % 2 == 0 else "odd",
    }


async def main() -> None:
    runner = AsyncBatchRunner(
        max_concurrency=2,
        results_file_path=Path("tasks.csv"),
    )
    results = await runner.map(
        describe,
        args_list=[{"value": 1}, {"value": 2}, {"value": 3}],
        task_ids=["one", "two", "three"],
    )
    # {"result": 2, "input": 1, "parity": "odd"}
    # {"result": 4, "input": 2, "parity": "even"}
    # {"result": 6, "input": 3, "parity": "odd"}


asyncio.run(main())
```


Each task appends a row to `tasks.csv`. The journal can be read without loading the whole file:

```python
for result in runner.read_results_csv():
    print(result["task_id"], result["success"], result["result"])
```

And will look like this (with different timestamps and durations):

```csv
task_id,start_time,end_time,duration_seconds,success,result,error,input,parity
one,2026-08-23T16:00:00+00:00,2026-08-23T16:00:00.100000+00:00,0.1000,True,2,,1,odd
two,2026-08-23T16:00:00+00:00,2026-08-23T16:00:00.100000+00:00,0.1000,True,4,,2,even
three,2026-08-23T16:00:00.100000+00:00,2026-08-23T16:00:00.200000+00:00,0.1000,True,6,,3,odd
```

Columns names are inferred from the dictionary returned by the function provided to the
`map` method. The function may also return plain text, which in this case will be
written in a  `result` column.

By default, if a task fails, `map` and `gather` returns/saves it's exception while
running the rest of the batch. 
