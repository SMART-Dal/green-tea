import json, os, sys, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent
from unsloth import FastLanguageModel
MODELS = {"Energy-SFT": os.environ["ESFT_MODEL"],  # merged Energy-SFT model (HF: green-tea-qwen2.5-coder-14b-energy-sft)
          "E-SFT+EDP": os.environ["GRPO_MODEL"]}  # GRPO adapter (HF: green-tea-qwen2.5-coder-14b-grpo)
N_SAMPLES = 5
PROMPT_FMT = ("This is an energy inefficient program we want to optimize to score 10/10.\n"
              "### Program:\n{code}\n\n"
              "### Energy Optimized Version with score 10/10:\n```cpp\n")
def extract_code(gen):
    if '```cpp' in gen: return gen.split('```cpp')[1].split('```')[0].strip()
    if '```' in gen: return gen.split('```')[0].strip()
    return gen.strip()
cases = json.load(open(str(HERE / os.environ.get("CASE_SET", "systems_stdin") / "cases.json")))
out = open(str(HERE / os.environ.get("CASE_SET", "systems_stdin") / "generations.jsonl"), "w")
for mname, path in MODELS.items():
    model, tok = FastLanguageModel.from_pretrained(model_name=path, max_seq_length=8192, dtype=torch.bfloat16)
    FastLanguageModel.for_inference(model)
    fence = tok.encode('```', add_special_tokens=False)[0]
    for cname, c in cases.items():
        code = "\n".join(c["code"].split("\n")[1:])  # drop the provenance comment line
        inputs = tok(PROMPT_FMT.format(code=code), return_tensors="pt").to(model.device)
        for k in range(N_SAMPLES):
            torch.manual_seed(k)
            with torch.no_grad():
                o = model.generate(**inputs, max_new_tokens=2048, temperature=0.2, do_sample=True,
                                   eos_token_id=[tok.eos_token_id, fence], pad_token_id=tok.pad_token_id, repetition_penalty=1.2)
            g = extract_code(tok.decode(o[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
            out.write(json.dumps(dict(model=mname, case=cname, sample=k, prompt_tokens=int(inputs["input_ids"].shape[1]), generated_code=g)) + "\n"); out.flush()
            print(mname, cname, k, inputs["input_ids"].shape[1], len(g), flush=True)
    del model; torch.cuda.empty_cache()
