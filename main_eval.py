import argparse
import json
import os
import numpy as np
import random
import torch
import time
from vllm import LLM, SamplingParams
from dataset.kk import KKProcessor
import datasets

def load_jsonl(file_path):
    records = []
    with open(file_path, "r") as file:
        for line in file:
            records.append(json.loads(line))
    return records

def write_jsonl(output_file, data):
    with open(output_file, "w") as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + "\n")

def init_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_eval_records(args, subject):
    if args.problem_type != "clean":
        records = datasets.load_dataset('K-and-K/perturbed-knights-and-knaves', data_files=f"{args.split}/{args.problem_type}/{subject}.jsonl")["train"] 
    else:
        records = datasets.load_dataset('K-and-K/knights-and-knaves', data_files=f"{args.split}/{subject}.jsonl")["train"]
    return records

def eval_subject(args, subject, llm, test_records, kk_proc, exist_result_records):
    """Evaluate one subject."""
    cors = []
    start_index = len(exist_result_records)
    print(f"Found existing {start_index} records in {subject}")
    for i in range(start_index):
        cors.append(exist_result_records[i]["correct"])
    eval_start_time = time.time()
    # Prepare all prompts
    prompts = []
    labels = []
    for i in range(start_index, len(test_records)):
        prompt, label = kk_proc.gen_test_prompt(
            args.ntrain, test_records, i, args.model
        )
        prompts.append(prompt)
        if i == start_index:
            print(f"Sample prompt:\n{prompt}")
        labels.append(label)
    # Get responses using vLLM
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_token,
    )
    responses = []
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        responses.extend([output.outputs[0].text for output in outputs])
    # Process results
    for i, (prompt, label, response) in enumerate(zip(prompts, labels, responses), start=start_index):
        cor, parsed_pred, reformat_gold_conditions = kk_proc._parse_cot_eval(response, label, args.model)
        if i % 1 == 0:
            print(f"\nPrompt {i}:{prompt}"
                    f"\nResponse {i}:{response}"
                    f"\nPrediction {i}:{parsed_pred}"
                    f"\nLabel {i}:{reformat_gold_conditions}"
                    f"\nCorrect {i}:{cor}")
        cors.append(cor)
        new_item = {
            'quiz': test_records[i]['quiz'], 
            'names': test_records[i]['names'],
            'solution': test_records[i]['solution'],
            'solution_text': test_records[i]['solution_text'],
            'solution_text_format': test_records[i]['solution_text_format'],
            'index': test_records[i]['index'],
            'predicts': parsed_pred,
            'labels': reformat_gold_conditions,
            'correct': cor,
            'response': response,
            'prompts': prompt,
        }
        exist_result_records.append(new_item)
    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    acc = np.mean(cors)
    cors = np.array(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print(f"Total evaluation time: {eval_time:.2f} seconds")
    return cors, acc, exist_result_records

def load_limited_test_records(args, subject, exist_result_records):
    """Load limited test records based on given arguments."""
    test_records = load_eval_records(args, subject)
    
    if args.limit is not None:
        test_records = test_records.select(range(min(args.limit, len(test_records))))
        if args.limit <= len(exist_result_records):
            return None # have finished  exp
    
    return test_records

def save_final_acc_results(all_cors, results, fname):
    """Process final results, calculate average accuracy, and save to file."""
    if all_cors:
        weighted_acc = np.mean(np.concatenate(all_cors))
        results["weighted_accuracy"] = weighted_acc
        print(f"Average accuracy: {weighted_acc:.3f}")
        with open(fname, "w") as f:
            json.dump(results, f)

def load_previous_acc_results(fname):
    """Load previous accuracy results."""
    acc_results = {"subject": {}}
    if os.path.isfile(fname):
        with open(fname, 'r', encoding='utf-8') as file:
            acc_results = json.load(file)
        print(f"Previous Results loaded successfully: {acc_results}")
    return acc_results

def get_subjects_to_eval(args):
    """Get subjects to evaluate."""
    subjects = []
    if args.split == "test":
        if args.eval_nppl == 0:
            subjects = [f"people{nppl}_num100" for nppl in range(2, 9)]
        else:
            subjects = [f"people{args.eval_nppl}_num100"]
    elif args.split == "train":
        if args.eval_nppl == 2:
            subjects = ["people2_num200"]
        elif args.eval_nppl > 2:
            subjects = [f"people{args.eval_nppl}_num1000"]
    return subjects

def main(args):
    model_short_name = "/".join(args.model.split("/")[-2:])
    prefix = os.path.join(
        os.path.join(args.save_dir, "{}_{}shot".format(
            model_short_name, args.ntrain))
    )
    args.config += f"_token{args.max_token}{('_cot' if args.cot else '')}" \
        f"_{args.split}{('_' + args.problem_type if args.problem_type != 'clean' else '')}"
    output_folder = os.path.join(prefix, args.config)
    acc_fname = os.path.join(prefix, f"result_{args.config}.json")
    os.makedirs(output_folder, exist_ok=True)
    print("args.config", args.config, "\nprefix", prefix, "\noutput_folder", output_folder)
    kk_proc = KKProcessor(cot=args.cot, no_linebreak=args.no_linebreak)
    subjects = get_subjects_to_eval(args)
    acc_results = load_previous_acc_results(acc_fname)
    # Initialize vLLM model
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.ngpus,  # 使用 ngpus 参数设置并行 GPU 数量
        max_model_len=args.max_token,
        gpu_memory_utilization=0.9,
    )
    all_cors = []
    for subject in subjects:
        result_outfile = os.path.join(output_folder, "{}.jsonl".format(subject))
        exist_result_records = load_jsonl(result_outfile) if os.path.exists(result_outfile) else []
        test_records = load_limited_test_records(args, subject, exist_result_records)
        if test_records is None:
            continue 
        cors, acc, result_records = eval_subject(args, subject, llm, test_records, kk_proc, exist_result_records)
        write_jsonl(result_outfile, result_records)
        all_cors.append(cors)
        acc_results["subject"][subject] = acc
    save_final_acc_results(all_cors, acc_results, acc_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for KK dataset")
    parser.add_argument("--ntrain", "-k", type=int, default=0, help="Number of training examples")
    parser.add_argument("--data_dir", "-d", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", "-s", type=str, default="result_qa", help="Save directory")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name or path")
    parser.add_argument("--max_token", type=int, default=1024, help="Maximum number of tokens")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--no_linebreak", action="store_true", help="Remove line breaks")
    parser.add_argument("--batch_size", type=int, default="auto", help="Batch size for VLLM")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Data split to use")
    parser.add_argument("--eval_nppl", type=int, default=0, help="Number of people to evaluate")
    parser.add_argument("--problem_type", type=str, default="clean", help="Problem perturbation type")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--config", type=str, default="default_config", help="Configuration string for saving results")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs for parallel inference")  # 添加 ngpus 参数
    args = parser.parse_args()
    init_seed()
    main(args)
