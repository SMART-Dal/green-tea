#!/usr/bin/env python3
"""Generate zero-shot and green-prompting baselines.
Base model variants: zero_shot, green_prompt (completion-style prompts)
Instruct model variants: zero_shot_instruct, green_prompt_instruct (ChatML via apply_chat_template)
Reads sft_pairs_test.jsonl, writes to data/sft_baseline_generations/{variant}/generations.jsonl.
"""
import json, argparse, torch
from pathlib import Path
from collections import defaultdict

# Base model: completion-style (model continues text after the cpp fence)
BASE_PROMPTS = {
    "zero_shot": (
        "Optimize the following C++ program.\n"
        "### Program:\n{code}\n\n"
        "### Optimized Version:\n```cpp\n"
    ),
    "green_prompt": (
        "Optimize the following C++ program to reduce energy consumption. "
        "Prefer energy-efficient algorithms and avoid unnecessary computation.\n"
        "### Program:\n{code}\n\n"
        "### Energy-Efficient Version:\n```cpp\n"
    ),
    "sft": (
        "This is an energy inefficient program we want to optimize to score 10/10.\n"
        "### Program:\n{code}\n\n"
        "### Energy Optimized Version with score 10/10:\n```cpp\n"
    ),
    "grpo": (
        "This is an energy inefficient program we want to optimize to score 10/10.\n"
        "### Program:\n{code}\n\n"
        "### Energy Optimized Version with score 10/10:\n```cpp\n"
    ),
}

# Instruct model: chat messages for apply_chat_template
INSTRUCT_MESSAGES = {
    "zero_shot_instruct": {
        "system": "You are an expert C++ programmer. Output only the energy-optimized C++ code inside a single ```cpp block, no explanation.",
        "user": "Generate an energy-optimized version of the following C++ program.\n\n```cpp\n{code}\n```",
    },
    "green_prompt_instruct": {
        "system": "You are an expert C++ programmer specializing in energy-efficient code. Output only the energy-optimized C++ code inside a single ```cpp block, no explanation.",
        "user": (
            "Generate an energy-optimized version of the following C++ program. "
            "Reduce energy consumption by using efficient algorithms, minimizing unnecessary computation, "
            "and preferring cache-friendly data structures.\n\n"
            "```cpp\n{code}\n```"
        ),
    },
}


def extract_code(text: str) -> str:
    if "```cpp" in text:
        return text.split("```cpp")[1].split("```")[0].strip()
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip() if text.count("```") >= 2 else text.split("```")[0].strip()
    return text.strip()


def generate_baselines(model_name: str, input_file: str, output_dir: str, variants: list,
                       max_new_tokens: int, num_samples: int = 1, temperature: float = 0.2,
                       top_p: float = 0.95, use_transformers: bool = False):
    if use_transformers:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, 'seen_tokens'):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, 'get_max_length'):
            DynamicCache.get_max_length = lambda self: self.get_seq_length()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=32768, dtype=torch.bfloat16
        )
    model.eval()
    print(f"Loaded {model_name}", flush=True)

    samples = [json.loads(l) for l in open(input_file)]
    by_baseline: dict = defaultdict(list)
    for s in samples:
        code = s.get("inefficient_code", s.get("baseline_code", ""))
        if code:
            by_baseline[code].append(s)
    print(f"{len(samples)} samples -> {len(by_baseline)} unique baselines", flush=True)
    print(f"Generating {num_samples} sample(s) per problem, temperature={temperature}", flush=True)

    fence_id = tokenizer.encode("```", add_special_tokens=False)[0]
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_id[0] if im_end_id else 151645

    for variant in variants:
        is_instruct = variant.endswith("_instruct")
        if is_instruct and variant not in INSTRUCT_MESSAGES:
            print(f"[{variant}] Unknown instruct variant, skipping", flush=True)
            continue
        if not is_instruct and variant not in BASE_PROMPTS:
            print(f"[{variant}] Unknown base variant, skipping", flush=True)
            continue

        out_dir = Path(output_dir) / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "generations.jsonl"

        completed: dict = defaultdict(int)
        if out_file.exists():
            for l in open(out_file):
                rec = json.loads(l)
                completed[rec["problem_id"]] += 1
            print(f"[{variant}] Resuming: {len(completed)} problems started", flush=True)

        written = 0
        with open(out_file, "a" if completed else "w") as f:
            for idx, (baseline_code, slist) in enumerate(by_baseline.items(), 1):
                if is_instruct:
                    tmpl = INSTRUCT_MESSAGES[variant]
                    messages = [
                        {"role": "system", "content": tmpl["system"]},
                        {"role": "user", "content": tmpl["user"].format(code=baseline_code)},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    stop_ids = [tokenizer.eos_token_id, im_end_id]
                else:
                    prompt = BASE_PROMPTS[variant].format(code=baseline_code)
                    stop_ids = [tokenizer.eos_token_id, fence_id]

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                if inputs["input_ids"].shape[1] + max_new_tokens > 32768:
                    continue

                for sample_idx in range(num_samples):
                    # Skip already-generated samples for each problem
                    skip_all = True
                    for s in slist:
                        pid = s.get("problem_id", "")
                        if completed.get(pid, 0) <= sample_idx:
                            skip_all = False
                            break
                    if skip_all:
                        continue

                    with torch.no_grad():
                        out = model.generate(
                            **inputs, max_new_tokens=max_new_tokens,
                            temperature=temperature, do_sample=True,
                            top_p=top_p,
                            eos_token_id=stop_ids,
                            pad_token_id=tokenizer.pad_token_id,
                            repetition_penalty=1.2,
                        )
                    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    code = extract_code(generated)
                    for s in slist:
                        pid = s.get("problem_id", "")
                        if completed.get(pid, 0) > sample_idx:
                            continue
                        f.write(json.dumps({
                            "problem_id": pid,
                            "sample_idx": sample_idx,
                            "baseline_code": baseline_code,
                            "generated_code": code,
                            "optimized_code": s.get("optimized_code", ""),
                        }) + "\n")
                        completed[pid] = completed.get(pid, 0) + 1
                        written += 1
                    f.flush()
                if idx % 50 == 0:
                    print(f"[{variant}] {idx}/{len(by_baseline)} -> {written} written", flush=True)
        print(f"[{variant}] Done: {written} outputs -> {out_file}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    p.add_argument("--input", default="data/sft_pairs_test.jsonl")
    p.add_argument("--output-dir", default="data/sft_baseline_generations")
    p.add_argument("--variants", default="zero_shot_instruct,green_prompt_instruct")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--num-samples", type=int, default=1, help="Samples per problem (for pass@k)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--use-transformers", action="store_true", help="Use transformers instead of unsloth (required for MoE models)")
    args = p.parse_args()
    generate_baselines(
        args.model, args.input, args.output_dir,
        [v.strip() for v in args.variants.split(",")],
        args.max_new_tokens, args.num_samples, args.temperature, args.top_p,
        use_transformers=args.use_transformers,
    )


if __name__ == "__main__":
    main()
