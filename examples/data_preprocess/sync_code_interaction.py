# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/sync_code")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "OpenCoder-LLM/opc-annealing-corpus"
    dataset = datasets.load_dataset(data_source, "algorithmic_corpus")

    train_dataset = dataset["train"]
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("text")
            
            lines: List[str] = question_raw.splitlines()
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
            question_raw = "\n".join(no_asserts)

            system_prompt = "# Generate either a comment as your thinking process before writing the next several lines of code, or directly write the next line of code.\n"
            question = system_prompt + question_raw

            answer_raw = ""
            solution = ""
            data = {
                "data_source": data_source,
                "prompt": question,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "interaction_kwargs": {
                        "name": "gsm8k",
                        "query": question,
                        "ground_truth": solution,
                    },
                },
            }
            return data

        return process_fn
    train_dataset = train_dataset.filter(lambda example: example["lang"]=="python")
    test_dataset = train_dataset.select(range(10))

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
