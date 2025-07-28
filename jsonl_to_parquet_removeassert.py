import re
import glob
import pandas as pd
from typing import List

def remove_unit_tests(text: str) -> str:
    """
    1. Drop any lines that start with `assert`
    2. Truncate everything after the last fenced code block (```...```)
    """
    lines: List[str] = text.splitlines()
    # 1) Remove assertion lines
    no_asserts = [
        line for line in lines
        if not re.match(r'^\s*assert\b', line)
    ]
    # 2) Find last code fence and cut off anything after it
    fence_idxs = [
        idx for idx, line in enumerate(no_asserts)
        if line.strip().startswith("```")
    ]
    if fence_idxs:
        last_fence = fence_idxs[-1]
        no_asserts = no_asserts[: last_fence + 1]
    return "\n".join(no_asserts)

# 1. Locate your JSONL files
jsonl_files = glob.glob("/mnt/task_runtime/opc-annealing-corpus/algorithmic_corpus/*.jsonl")

# 2. Read, filter, clean, and collect
dfs = []
for i, fn in enumerate(jsonl_files):
    df = pd.read_json(fn, lines=True)
    # keep only Python examples
    df = df[df['lang'] == 'python']
    # apply test‐removal to the column that holds code+tests
    df['text'] = df['text'].map(remove_unit_tests)
    if i == 0:
        df.to_parquet(
            "/mnt/task_runtime/dummytest.parquet",
            index=False
        )
    dfs.append(df)

# 3. Concatenate all DataFrames
all_data = pd.concat(dfs, ignore_index=True)
print(all_data, "examples after cleaning")
# 4. Write out a single Parquet
all_data.to_parquet(
    "/mnt/task_runtime/sync_python_removeassert.parquet",
    index=False
)