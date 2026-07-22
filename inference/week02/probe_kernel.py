import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-3B-Instruct"
batch_size = int(os.environ.get("BATCH_SIZE", "1"))

if batch_size < 1:
    raise ValueError(f"BATCH_SIZE must be >= 1, got {batch_size}")

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()

prompt = "Hello world"
prompts = [prompt] * batch_size
inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")

print(
    f"Batch size: {batch_size}, prompt tokens: {inputs.input_ids.shape[1]}",
    flush=True,
)

# warmup
with torch.no_grad():
    for _ in range(3):
        _ = model.generate(**inputs, max_new_tokens=4)
torch.cuda.synchronize()

# profile a few decode steps
with torch.no_grad():
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        out = model(**inputs, use_cache=True)
        past = out.past_key_values
        for _ in range(5):
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
