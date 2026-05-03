# decode_benchmark_ncu.py
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-3B-Instruct"

warmup_rounds = int(os.environ.get("WARMUP_ROUNDS", "2"))
warmup_decode_steps = int(os.environ.get("WARMUP_DECODE_STEPS", "4"))
profile_decode_steps = int(os.environ.get("PROFILE_DECODE_STEPS", "5"))
batch_size = int(os.environ.get("BATCH_SIZE", "1"))

if batch_size < 1:
    raise ValueError(f"BATCH_SIZE must be >= 1, got {batch_size}")

print("Loading tokenizer", flush=True)
tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

print("Loading model", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="cuda"
)
model.eval()

prompt = "Explain memory bandwidth bottleneck in LLM decoding briefly."
prompts = [prompt] * batch_size
inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")

print(
    f"Batch size: {batch_size}, prompt tokens: {inputs.input_ids.shape[1]}",
    flush=True,
)

print("Warmup", flush=True)
with torch.no_grad():
    for _ in range(warmup_rounds):
        out = model(**inputs, use_cache=True)
        past = out.past_key_values
        for _ in range(warmup_decode_steps):
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values

torch.cuda.synchronize()

print("Profiling decode steps", flush=True)
with torch.no_grad():
    out = model(**inputs, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()  # NCU range start

    for _ in range(profile_decode_steps):
        out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()  # NCU range end

print("Done")
