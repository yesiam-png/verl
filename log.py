from datasets import load_dataset
#from vllm import LLM, SamplingParams
import sglang as sgl
from transformers import AutoTokenizer
from huggingface_hub import HfApi
import json
import os
from tqdm import tqdm

def run_inference(model_name, prompts, output_file, max_tokens=22800, temperature=0.0):
    """
    Runs inference for a list of prompts using vLLM, counts response tokens,
    and saves outputs to a JSONL file.
    """
    #engine = LLM(model=model_name, trust_remote_code=True)
    engine = sgl.Engine(model_path=model_name,)
#    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, skip_special_tokens=False)  # stop=["####"],
    sampling_params = {"temperature":temperature, "skip_special_tokens": False}  # stop=["####"],

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    with open(output_file, "w") as f:
        for prompt in tqdm(prompts, desc=f"Inferencing with {model_name}"):
           # chat_prompt = prompt + "\n\nPresent your Python code within \n```python\nYour code\n```\nbelow.\n\n"


            chat_prompt =  "Think step by step and output the final answer after `####`.\n\n" + prompt + "\n\n"

           # message = [{"role": "user", "content": chat_prompt}]
           # chat_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

#            outputs = engine.generate(chat_prompt, sampling_params=sampling_params)
            outputs = engine.generate(chat_prompt, sampling_params=sampling_params)
#            completion = outputs[0].outputs[0].text
            completion = outputs['text']
            print("come", completion)
          #  print("zsa", outputs[0].outputs[0].token_ids)
            token_ids = tokenizer(completion).input_ids
            num_tokens = len(token_ids)

            record = {
                "chat_prompt": chat_prompt,
                "completion": completion,
                "num_tokens": num_tokens
            }
            f.write(json.dumps(record) + "\n")

def upload_to_hf(repo_name, file_path, token):
    """
    Creates or updates a dataset repo on Hugging Face and uploads a file.
    """
    api = HfApi(token=token)
    # Create the dataset repo if it doesn't exist
    api.create_repo(
        repo_id=repo_name,
        repo_type="dataset",
        private=False,
        exist_ok=True
    )
    # Upload the JSONL file
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_name,
        repo_type="dataset",
        token=token
    )

def main():
    # Hugging Face authentication token in env var HF_HUB_TOKEN
    USERNAME = os.environ.get('HF_USERNAME')
    hf_token = os.environ.get("HF_HUB_TOKEN")
    if hf_token is None:
        raise EnvironmentError("Please set your HF_HUB_TOKEN environment variable")

    # Load datasets
    dataset1 = load_dataset("openai/gsm8k", 'main', split="train[:10]")
   # dataset2 = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5", split="train[:15]")
    prompts = dataset1["question"]
    # List of model checkpoints
    models = ["/mnt/task_wrapper/user_output/artifacts/checkpoints/gsm8k_async_rl/nostdnomean-500len-newprompt-03up-llama-3b_function_rm-gsm8k-async-sgl-multi-w-tool-verify-n16-4cards/global_step_105/actor/huggingface",
   # "/mnt/task_runtime/global_step_60/actor/huggingface",
    #"/mnt/task_wrapper/user_output/artifacts/checkpoints/gsm8k_async_rl/qwen2.5-3b_function_rm-gsm8k-async-sgl-multi-w-tool-verify-n16-4cards/global_step_60/actor/huggingface",#"USERNAME/Llama-3.2-1B", "USERNAME/code_cpt"
      #  "/mnt/task_wrapper/user_output/artifacts/checkpoints/deepcoder/llama1b-cpt-12k/actor/global_step_10",
      #  "/mnt/task_wrapper/user_output/artifacts/checkpoints/deepcoder/llama1b-cpt-12k/actor/global_step_50",
      #  "/mnt/task_wrapper/user_output/artifacts/checkpoints/deepcoder/llama1b-cpt-12k/actor/global_step_70",
      #  "/mnt/task_wrapper/user_output/artifacts/checkpoints/deepcoder/llama1b-cpt-12k/actor/global_step_100",
      #  "/mnt/task_wrapper/user_output/artifacts/checkpoints/deepcoder/llama1b-cpt-12k/actor/global_step_200",
      #  "/mnt/task_wrapper/user_output/artifacts/checkpoints/deepcoder/llama1b-cpt-12k/actor/global_step_300",
    ]

    for model_path in models:
        step = model_path[-25:-18]#.split("step_")[-1]
        output_file = f"./llamawhy.jsonl"
        repo_name = f"{USERNAME}/llamawhy"
        
        # Run inference and save locally
        run_inference(model_path, prompts, output_file)

        # Upload results to Hugging Face
        upload_to_hf(repo_name, output_file, hf_token)
        print(f"Uploaded {output_file} to HF dataset repo {repo_name}")

if __name__ == "__main__":
    main()
