import pandas as pd
import glob

# 1. Find your JSONL files
jsonl_files = glob.glob("/mnt/task_runtime/opc-annealing-corpus/synthetic_qa/*.jsonl")

# 2. Read each one into a DataFrame
dfs = []
for fn in jsonl_files:
    # read_json with lines=True handles JSON Lines
    df = pd.read_json(fn, lines=True)
    df = df[df['program_lang'] == 'python']
    dfs.append(df)

# 3. Concatenate them all
all_data = pd.concat(dfs, ignore_index=True)

# 4. Write out a single Parquet file
all_data.to_parquet("/mnt/task_runtime/opencoder.parquet", index=False)

"""
# — or — write out in multiple Parquet “row-group” files of ~100 MB each:
# this splits the DataFrame into chunks of N rows
chunk_size = 200_000
for i in range(0, len(all_data), chunk_size):
    chunk = all_data.iloc[i : i + chunk_size]
    chunk.to_parquet(f"output/all_data_part{i//chunk_size:03d}.parquet", index=False)
"""