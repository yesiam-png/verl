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
    parser.add_argument("--local_dir", default="~/data/comments")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "OpenCoder-LLM/opc-annealing-corpus"
    dataset = datasets.load_dataset(data_source, "synthetic_code_snippet")

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
        lines = code_text.splitlines(keepends=False) # no \n in it!!!
        if not lines:
            return []
     #   return lines

        result_chunks = []
        current_chunk = []
        # This will track the type of the current chunk: 'multiline', 'single_line', 'code', or 'blank'
        chunk_type = None 
        in_multiline_comment = False

        for line in lines:
            stripped = line.strip()
            current_line_type = 'code'  # Assume it's code by default

            # 1. Determine the exact type of the current line
            if in_multiline_comment:
                current_line_type = 'multiline'
                if any(d in stripped for d in block_delimiters):
                    in_multiline_comment = False
            elif not stripped:
                current_line_type = 'blank'
            elif stripped.startswith(comment_char):
                current_line_type = 'single_line'
            elif any(stripped.startswith(d) for d in block_delimiters):
                current_line_type = 'multiline'
                delimiter = next(d for d in block_delimiters if stripped.startswith(d))
                if stripped.count(delimiter) < 2:
                    in_multiline_comment = True

            # 2. If the line type has changed, finalize the previous chunk and start a new one.
            # We will group blank lines with whatever chunk came before them.
            if current_line_type != chunk_type and current_line_type != 'blank':
                if current_chunk:
                    result_chunks.append("\n".join(current_chunk))
                current_chunk = []
                chunk_type = current_line_type
            
            # 3. Add the current line to its chunk
            current_chunk.append(line)

        # Add the very last chunk after the loop finishes
        if current_chunk:
            result_chunks.append("\n".join(current_chunk))

        return result_chunks

    import re
    from typing import List  # you already annotate with List below

    def remove_examples_in_docstrings(src: str) -> str:
        """
        Inside every triple-quoted string, drop everything starting at a line
        'Example:' / 'Examples:' through the end of that docstring.
        Also prunes stray doctest lines (>>> / ...).
        """
        triple = re.compile(r'(?s)(?P<q>"""|\'\'\')(?P<body>.*?)(?P=q)')

        def _rewrite(m: re.Match) -> str:
            body = m.group('body')
            # Remove from the Examples? header to end-of-docstring
            body = re.sub(r'(?ms)^\s*Example(?:s)?\s*:\s*\n.*\Z', '', body)
            # (Optional) remove doctest prompts if no explicit header was present
            body = re.sub(r'(?m)^\s*(>>>|\.\.\.)\s.*\n?', '', body)
            return f"{m.group('q')}{body}{m.group('q')}"

        return triple.sub(_rewrite, src)


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("text")
            question_raw = remove_examples_in_docstrings(question_raw)

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
            try:
                while split_lines[0].strip() == "":
                    split_lines = split_lines[1:]
                split_lines[0] = split_lines[0] + "\n" + split_lines[1]
                del split_lines[1]
            except IndexError:
                print(question, "zz")
                split_lines == []
             #   raise
          #  split_lines[0] = split_lines[0] + "\n" + split_lines[1]
          #  del split_lines[1]
          #  split_lines[0] = split_lines[0] + "\n" + split_lines[1]
          #  del split_lines[1]

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
    train_dataset = train_dataset.filter(lambda example: len(example["text"].splitlines()) > 1)

    #train_dataset = train_dataset.select(range(len(train_dataset) // 6))
    test_dataset = train_dataset.select(range(10))

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset = train_dataset.filter(lambda example: len(example["split_lines"])>1)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
