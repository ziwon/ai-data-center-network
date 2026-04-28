import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

prompt = "Explain memory bandwidth bottleneck in LLM decoding."
inputs = tok(prompt, return_tensors="pt").to("cuda")

# warm-up
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=8)
torch.cuda.synchronize()

timestamps = []

with torch.no_grad():
    past_key_values = None
    input_ids = inputs.input_ids

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for step in range(128):
        out = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        torch.cuda.synchronize()
        timestamps.append(time.perf_counter())

        input_ids = next_token

        if tok.eos_token_id is not None and next_token.item() == tok.eos_token_id:
            break

ttft = timestamps[0] - t_start
tpots = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

tpots_ms = np.array(tpots) * 1000

print(f"Generated tokens: {len(timestamps)}")
print(f"TTFT: {ttft * 1000:.1f} ms")

if len(tpots_ms) > 0:
    print(f"TPOT mean: {tpots_ms.mean():.1f} ms")
    print(f"TPOT p50: {np.percentile(tpots_ms, 50):.1f} ms")
    print(f"TPOT p95: {np.percentile(tpots_ms, 95):.1f} ms")
    print(f"TPOT p99: {np.percentile(tpots_ms, 99):.1f} ms")