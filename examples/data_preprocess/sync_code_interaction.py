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

    def split_on_code_line(code_text: str, comment_char: str = '#', block_delimiters=('"""', "'''")) -> list[str]:
        """
        Splits a block of text into a list of strings, accounting for multi-line
        comment blocks. Each string ends with the first line of code it encounters.

        Args:
            code_text: The multi-line string of code to split.
            comment_char: The character that indicates a single-line comment.
            block_delimiters: A tuple of strings that start and end block comments.

        Returns:
            A list of strings, where each element is a block of comments,
            blank lines, and the single line of code that follows it.
        """
        lines = code_text.splitlines(keepends=True) # no \n in it!!!
        if not lines:
            return []

        result_chunks = []
        current_chunk = []
        in_multiline_comment = False

        for line in lines:
            stripped = line.strip()
            is_this_line_code = True  # Assume it's code until proven otherwise

            # 1. Check if we are currently inside a multi-line comment
            if in_multiline_comment:
                is_this_line_code = False
                # Check if this line ends the block
                if any(d in stripped for d in block_delimiters):
                    in_multiline_comment = False
            
            # 2. Check for other non-code line types
            elif not stripped:  # An empty line
                is_this_line_code = False
            elif stripped.startswith(comment_char):  # A single-line comment
                is_this_line_code = False
            elif any(stripped.startswith(d) for d in block_delimiters):  # Starts a multi-line comment
                is_this_line_code = False
                # Check if the block also closes on the same line. If not, enter multi-line mode.
                delimiter = next(d for d in block_delimiters if stripped.startswith(d))
                if stripped.count(delimiter) < 2:
                    in_multiline_comment = True

            # 3. Append the line and split if it was determined to be code
            current_chunk.append(line)
            if is_this_line_code:
                result_chunks.append("\n".join(current_chunk))
                current_chunk = []

        # Add any trailing lines (e.g., final comments/blank lines)
        if current_chunk:
            result_chunks.append("\n".join(current_chunk))

        return result_chunks


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
                ids for ids, line in enumerate(no_asserts)
                if line.strip().startswith("```")
            ]
            if fence_idxs:
                last_fence = fence_idxs[-1]
                no_asserts = no_asserts[: last_fence + 1]
            question_raw = "\n".join(no_asserts)
            system_prompt = ""

      #      system_prompt = "For each upcoming section of code, either provide a concise comment explaining it, OR directly skip to the next line."
#            system_prompt = "After each <eol>, either provide a concise comment explaining the purpose and logic of the upcoming section of code, OR directly skip to the next line."
          #  system_prompt = "Generate either a comment to explain the next several lines of code, or skip directly to the next line."
            question = system_prompt + question_raw

            split_lines = split_on_code_line(question)
            split_lines[0] = split_lines[0] + split_lines[1]
            del split_lines[1]
            split_lines[0] = split_lines[0] + split_lines[1]
            del split_lines[1]

            answer_raw = ""
            solution = ""
            data = {
                "data_source": data_source,
                "prompt": question,
                "split_lines": split_lines,
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

    #train_dataset = train_dataset.select(range(len(train_dataset) // 6))
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
