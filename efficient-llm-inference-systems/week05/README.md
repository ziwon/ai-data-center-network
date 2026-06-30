# Week 5 — Speculative Decoding

These notes expand Chapter 5 of the book. This version keeps the source chapter's core topic, speculative decoding, while connecting it to the Week 1-4 measurements: low-batch decode is launch-overhead-bound and memory-bandwidth-bound, KV cache has real memory cost, and kernel quality can matter more than bit width.

The main systems question this week is: **when is it worth spending extra draft-model memory to reduce decode latency?** The VRS verifier is a useful concrete case because it is latency-sensitive, low-concurrency, and produces structured output.

## 5.1 Learning Goals

By the end of this week, you should be able to:

1. Explain the core mathematical identity behind speculative decoding: **rejection sampling preserves the target model distribution**.
2. Derive how acceptance rate α determines speedup, and remember typical α ranges for different workloads.
3. Compare the trade-offs among draft+target speculative decoding, Medusa multi-head decoding, and EAGLE hidden-state-aware draft decoding.
4. Decide whether the Week 4 measurement, Marlin AWQ-INT4 = 2.14x speedup, stacks multiplicatively or additively with speculative decoding: SD on top of W4.
5. Analyze the structured-output behavior of a VRS verifier, using Cosmos-Reason2-2B as the baseline, and decide whether speculative decoding is worth applying.

## 5.2 Prerequisite Check

You should already know:

- The fact from the Week 2 NCU measurement: **decode was launch-overhead-bound and memory-bandwidth-bound**.
- The fact from the Week 3 KV cache accounting: **draft + target means two sets of KV cache**.
- The lesson from Week 4 measurement: **kernel quality can matter more than bit width**.
- The VRS architecture: a fast path (YOLOE) and a slow path (VLM verifier), where **the verifier only processes stable candidates**.

These facts feed directly into the cost/benefit analysis for speculative decoding. **The essence of speculative decoding is trading extra KV memory for lower latency.** The VRS verifier is close to an ideal use case for that trade.

---

## 5.3 Core Concept: Draft and Verify

### 5.3.1 Basic Algorithm

```
1. Use a small draft model M_d to autoregressively generate K candidate tokens.
   - This is fast: small model, K forward passes.

2. Use the large target model M_t to verify these K+1 positions in one pass.
   - prompt + draft_1
   - prompt + draft_1 + draft_2
   - ...
   - prompt + ... + draft_K
   - This is a single parallel forward pass.

3. Accept or reject each token with rejection sampling:
   - if p_t(x) >= p_d(x): accept
   - otherwise accept with probability p_t(x) / p_d(x)
   - stop at the first rejected position

4. Sample the correction token from the target distribution.
```

**Core properties:**

- **The output distribution is exactly the target model's distribution.**
- There is no quality loss. For T=0 greedy decoding, the sequence is exactly the same. For T>0 sampling, the distribution is the same.
- Higher acceptance rate makes the method more efficient.

### 5.3.2 Why It Works: Using Asymmetry

Compare decoding K tokens one token at a time with verifying K positions in one forward pass:

```
Decode K tokens autoregressively: K x (weight_load_time + compute)
                                = K x 12 ms (3B model example)
                                = 60 ms (K=5)

K positions in parallel:          1 x (weight_load_time + K x compute)
                                = 12 ms + epsilon
                                ~= 14 ms
```

**Weights are loaded once, while compute scales with K.** Because decode is memory-bound, processing K positions does not increase time much. This is the mechanical basis of speculative decoding.

This is a direct application of the Week 2 NCU result. Batch=1 attention had SM Busy 1.13%, while batch=32 had SM Busy 31.5%. **At batch=1, the GPU is mostly idle, so adding K positions is close to free.**

### 5.3.3 Speedup Formula

```
Expected accepted tokens per verification: 1 + α + α² + ... + α^K
                                          = (1 - α^(K+1)) / (1 - α)

Verification cost: 1 target forward pass = T_t
Draft cost, sequential: K x T_d

Speedup = (1 - α^(K+1)) / (1 - α) x T_t / (T_t + K x T_d)
```

Approximation, assuming constant α and large K:

```
Speedup ~= 1 / (1 - α)
```

| α | K=4 expected speedup | K=8 expected speedup |
|---|---|---|
| 0.5 | 1.94x | 2.0x |
| 0.7 | 2.6x | 3.2x |
| 0.8 | 3.4x | 4.5x |
| 0.9 | 4.7x | 7.2x |

**The higher α is, the more valuable a larger K becomes.** If α is low, going beyond K=4 usually does not help.

### 5.3.4 What Acceptance Rate Depends On

| Factor | Effect on α |
|---|---|
| Task type | Code: 0.8-0.9; factual QA: 0.7-0.8; creative writing: 0.4-0.6 |
| Draft-target alignment | Same-family draft and target, such as Llama draft for Llama target, gives higher α |
| Draft size | Too small means poor quality and low α; too large means high draft cost |
| Sampling temperature | T=0 greedy gives higher α; T=1.0 spreads probability mass and lowers α |
| Speculative window K | Larger K causes compounding rejection, so marginal α falls |

---

## 5.4 Variants of Speculative Decoding

### 5.4.1 Vanilla Speculative Decoding: Separate Draft Model

This is the most direct form. For example, use Qwen2.5-VL-2B as the draft and Qwen2.5-VL-7B as the target.

**Advantages:**

- Plug-and-play, with no target-model modification.
- Draft quality can be improved independently.
- Production-ready in systems such as vLLM and TensorRT-LLM.

**Disadvantages:**

- Both models must be resident in memory.
- Two KV caches and two model loads are expensive on edge devices.
- Draft-target distribution mismatch can lower acceptance rate.

### 5.4.2 Medusa: Multi-Head Speculative Decoding

Medusa avoids a separate draft model by attaching additional prediction heads to the target model's final hidden state:

```
From the target model's final hidden state h_t:
  Head 0 (original): predict token_t+1
  Head 1 (added):    predict token_t+2
  Head 2 (added):    predict token_t+3
  ...
  Head K-1:          predict token_t+K
```

**Advantages:**

- No extra model memory. The heads are very small, often tens of MB.
- Only one KV cache is needed.
- Training is relatively cheap: freeze the base model and train only the heads.

**Disadvantages:**

- α is often lower than vanilla speculative decoding because each head sees only the hidden state, not sequential conditioning.
- Production-grade integration takes additional work.
- It needs a custom training or fine-tuning pipeline.

### 5.4.3 EAGLE / EAGLE-2: Hidden-State-Aware Draft

EAGLE is a separate draft model, but it receives the target model's hidden state as input:

```
EAGLE draft model input:
  - previous token embedding
  - previous target model hidden state (feature)

Draft model = small autoregressive model with two inputs
```

This is why EAGLE can produce higher-quality drafts than Medusa: **the draft directly uses the target model's internal representation**.

**EAGLE-2** adds dynamic draft tree expansion. Branches with high acceptance are expanded deeper, while low-acceptance branches stay shallow. This explores more possible continuations under the same K budget.

Paper-level benchmark ranges:

- Vanilla speculative decoding: 2.5-3.0x speedup
- Medusa: 2.0-2.5x
- EAGLE: 3.0-3.5x
- EAGLE-2: 3.5-4.5x, and 5x+ on some workloads

### 5.4.4 One-Line Comparison

| Method | Extra memory | Typical α | Integration complexity |
|---|---|---|---|
| Vanilla SD | Large: full draft model | 0.7-0.8 | Low: production-ready |
| Medusa | Very small | 0.6-0.7 | Medium: training required |
| EAGLE-2 | Small: small draft | 0.8-0.9 | Medium-high |

---

## 5.5 When Speculative Decoding Does Not Help

### 5.5.1 High-Batch Serving

In the Week 1 batch sweep, the GPU was already saturated around batch=32-64. Speculative decoding is a **single-stream or low-batch latency optimization**. If the GPU is already full of work, parallel verification just adds more GPU work and can reduce throughput.

**Rule of thumb:** if GPU utilization, such as NCU SM Busy, is above 30%, speculative decoding is unlikely to help.

### 5.5.2 Memory-Tight Scenarios

Draft + target means two models in memory. Revisit the Week 4 Orin accounting:

```
AGX Orin 64GB:
- Target: 7B BF16 = 14 GB, or INT4 = 3.5 GB
- Draft candidates:
  - 1B BF16 = 2 GB -> extra 14% memory
  - 1B INT4 = 0.5 GB -> extra 3.5%

KV cache, two sets:
- 7B target: 56 KB/token
- 1B draft: 8 KB/token, small model with GQA
- Combined: about 64 KB/token, a 12% increase

Total extra memory: about 15-18%
```

This can be manageable, but it is still a real budget item.

### 5.5.3 Low-Acceptance Workloads

Free-form generation, such as captioning, creative writing, or multi-turn chat, often has α around 0.4-0.6. Speedup can stay near 1.5x. In this regime, **draft computation eats into the speculative decoding benefit**, and the acceleration may not be worth the added complexity.

### 5.5.4 Decision Frame

| Condition | SD decision |
|---|---|
| Single-stream or low-batch, below 8 | Consider |
| Expected α > 0.7 | Yes |
| Expected α < 0.5 | No |
| GPU SM Busy > 30% | No; already saturated |
| Edge + tight memory | Consider EAGLE or Medusa; vanilla may be expensive |
| Cloud + high-batch production | No; continuous batching is usually more effective |

---

## 5.6 Interactions with Other Optimizations

### 5.6.1 SD + Quantization: Multiplicative or Additive?

Week 4 result: Marlin AWQ-INT4 = 2.14x speedup. Typical speculative decoding speedup: 2.5x, assuming α=0.7-0.8.

**Do the two speedups multiply? Is it 2.14 x 2.5 = 5.35x?**

Answer: **almost multiplicative, but not perfectly.** Reasons:

1. **Target model verification can be quantized**, so the 2.14x improvement applies to the target path.
2. **The draft model can also be quantized**, so the draft path gets faster too.
3. **Acceptance rate can be affected by quantization.** A quantized model's distribution is slightly different, so α can fall, often by 5-10%.
4. **KV cache does not shrink automatically**, so memory benefits are not a pure product.

Empirical pattern from papers and benchmarks:

- W4 only: 2x speedup
- SD only: 2.5x speedup
- W4 + SD: 4.5x speedup, about 90% of the naive 5x expectation

**Speculative decoding and quantization can stack.** For the VRS verifier, AWQ-INT4 + EAGLE-2 SD can plausibly give about 4-5x speedup. This is the strongest combination for reducing baseline BF16 verifier latency.

### 5.6.2 SD + Continuous Batching: Preview of Week 6

Continuous batching is a high-throughput serving technique. Speculative decoding is a low-batch latency technique. They can conflict:

- High batch + SD: the methods can damage each other.
- Low batch + SD: SD works well.
- Decision: split by workload. Use an SD path only for latency-sensitive requests.

### 5.6.3 SD + Disaggregated Serving: Preview of Week 8

Disaggregated serving separates prefill nodes and decode nodes. Speculative decoding optimizes decode. **The prefill node is mostly unrelated to SD.** Apply speculative decoding on the decode node.

---

## 5.7 Reading

### Required

- Full Chapter 5 of the source book

### Recommended Papers

1. **Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding"**: Google paper; mathematical foundation for vanilla speculative decoding.
2. **Chen et al. 2023, "Accelerating Large Language Model Decoding with Speculative Sampling"**: DeepMind paper; independent discovery around the same time, with rejection sampling proof.
3. **Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"**: original Medusa paper.
4. **Li et al. 2024, "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"**: EAGLE-2 and current high-performance draft-tree approach.
5. **NVIDIA TensorRT-LLM speculative decoding documentation**: production integration guide.

The Leviathan paper is the best starting point for the mathematical foundation. It is short and clear.

---

## 5.8 Labs

### Lab 1: Implement a Toy Speculative Decoder

Implement the simplest speculative decoding loop to understand the mechanism:

```python
# week05/toy_sd.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

target_id = "Qwen/Qwen2.5-3B-Instruct"
draft_id = "Qwen/Qwen2.5-0.5B-Instruct"  # same family, 6x smaller

tok = AutoTokenizer.from_pretrained(target_id)
target = AutoModelForCausalLM.from_pretrained(
    target_id, dtype=torch.bfloat16, device_map="cuda"
)
draft = AutoModelForCausalLM.from_pretrained(
    draft_id, dtype=torch.bfloat16, device_map="cuda"
)
target.eval()
draft.eval()

def speculative_decode_step(prompt_ids, draft_past, target_past, K=5):
    """One SD step: generate K draft tokens, verify with target."""

    # Step 1: Draft K tokens autoregressively
    draft_tokens = []
    draft_probs = []
    current_input = prompt_ids[:, -1:]

    with torch.no_grad():
        for k in range(K):
            out = draft(input_ids=current_input, past_key_values=draft_past, use_cache=True)
            draft_past = out.past_key_values
            logits = out.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            tok_id = probs.argmax(dim=-1, keepdim=True)
            draft_tokens.append(tok_id)
            draft_probs.append(probs.gather(-1, tok_id))
            current_input = tok_id

    # Step 2: Target verifies K tokens in parallel
    draft_seq = torch.cat(draft_tokens, dim=-1)
    target_input = torch.cat([prompt_ids[:, -1:], draft_seq], dim=-1)

    with torch.no_grad():
        out = target(input_ids=target_input, past_key_values=target_past, use_cache=True)
        target_logits = out.logits
        target_probs = torch.softmax(target_logits, dim=-1)

    # Step 3: Rejection sampling
    accepted = []
    for k in range(K):
        draft_p = draft_probs[k].item()
        target_p = target_probs[0, k, draft_tokens[k].item()].item()

        if target_p >= draft_p:
            accepted.append(draft_tokens[k])
        else:
            if torch.rand(1).item() < target_p / draft_p:
                accepted.append(draft_tokens[k])
            else:
                corrected = torch.clamp(target_probs[0, k] - draft_probs[k], min=0)
                corrected = corrected / corrected.sum()
                resampled = torch.multinomial(corrected, 1)
                accepted.append(resampled.unsqueeze(0))
                break

    return accepted, len(accepted)

prompt = "The capital of France is"
input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    target_out = target(input_ids=input_ids, use_cache=True)
    draft_out = draft(input_ids=input_ids, use_cache=True)
    target_past = target_out.past_key_values
    draft_past = draft_out.past_key_values

total_accepted = 0
total_proposed = 0
K = 5
for step in range(10):
    accepted, n_accept = speculative_decode_step(input_ids, draft_past, target_past, K=K)
    total_accepted += n_accept
    total_proposed += K
    input_ids = torch.cat([input_ids] + accepted, dim=-1)

print(f"Acceptance rate: {total_accepted/total_proposed:.2f}")
print(f"Generated tokens: {input_ids.shape[1]}")
```

**Important:** this code is for understanding the mechanism. KV cache management is simplified, so it is not production-grade. vLLM and TensorRT-LLM handle this correctly.

### Lab 2: Measure α for VRS Verifier Output

This is the main experiment for this week. Quantitatively decide whether the VRS verifier is a good fit for speculative decoding.

Typical VRS verifier output with Cosmos-Reason2-2B:

```json
{
  "verdict": "yes" or "no",
  "confidence": 0.85,
  "rationale": "Person on floor in distressed pose, no movement for 8 seconds."
}
```

**Hypothesis:** verifier output is highly structured, so α should be high, possibly 0.8+. If true, speculative decoding is very valuable.

```python
# week05/vrs_verifier_alpha.py
# Measure α with the prompt template and sample inputs used by the VRS verifier.

# Approximation of the VRS verifier prompt template
VERIFIER_PROMPT_TEMPLATE = """You are a CCTV alert verifier.
Given the event candidate, decide if this is a real event.

Event type: {event_type}
Detector confidence: {detector_conf}
Bounding box: {bbox}
Context window: {context_frames} frames

Output format:
{{
  "verdict": "yes" or "no",
  "confidence": <0.0 to 1.0>,
  "rationale": "<short explanation in 1-2 sentences>"
}}
"""

# Test cases from VRS watch-policy event types
test_cases = [
    {"event_type": "falldown", "detector_conf": 0.87, "bbox": "[120, 340, 280, 520]"},
    {"event_type": "fire", "detector_conf": 0.92, "bbox": "[400, 200, 600, 480]"},
    {"event_type": "smoke", "detector_conf": 0.78, "bbox": "[100, 100, 800, 400]"},
    {"event_type": "weapon", "detector_conf": 0.94, "bbox": "[50, 250, 180, 380]"},
    # Add more cases.
]

# Use the toy SD code to measure α for each test case.
# Compare against free-form caption prompts as the control.
free_form_prompts = [
    "Describe what you see in this scene.",
    "Write a story about the event.",
]

# Analyze:
# - VRS structured-output α
# - free-form-output α
# - whether the gap is large
```

Expected pattern:

- **JSON schema tokens, such as `"verdict":`, `"confidence":`, and `"rationale":`:** α ~= 0.95, nearly deterministic.
- **Verdict value, `"yes"` or `"no"`:** α ~= 0.8, because there are only two likely values and a good draft often matches.
- **Confidence number:** α ~= 0.6, because exact continuous values are hard to match.
- **Rationale text:** α ~= 0.5-0.7, because it is partly free-form.

**Overall weighted average α is expected to be around 0.7-0.8.** That implies about 2.5-3.5x speculative decoding speedup.

If the measurement matches the hypothesis, speculative decoding should be applied to the VRS verifier. If the confidence-number section has very low α, the structured output schema itself can be redesigned to be more SD-friendly. For example, replace continuous confidence with discrete categories such as `"low"`, `"med"`, and `"high"`.

This measurement directly becomes a VRS system improvement proposal.

### Lab 3: Measure Production-Grade vLLM Speculative Decoding

Benchmark speculative decoding in a production-style runtime:

```bash
pip install vllm

# Baseline, no speculative decoding
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8000

# With speculative decoding, draft + target
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --speculative-model Qwen/Qwen2.5-0.5B-Instruct \
    --num-speculative-tokens 5 \
    --port 8001
```

Send the same prompts to both endpoints and measure latency:

```python
# week05/vllm_sd_bench.py
import requests
import time

# VRS-style verifier prompts
vrs_prompts = [
    "Event: falldown, conf 0.87. Output JSON verdict.",
    "Event: fire, conf 0.92. Output JSON verdict.",
    "Event: smoke, conf 0.78. Output JSON verdict.",
]

# Free-form prompts as the control
free_prompts = [
    "Describe the scene in 5 sentences.",
    "Write a short story.",
]

for prompts, label in [(vrs_prompts, "VRS-style"), (free_prompts, "Free-form")]:
    print(f"\n=== {label} ===")
    for prompt in prompts:
        for port, name in [(8000, "baseline"), (8001, "with SD")]:
            t = time.time()
            resp = requests.post(f"http://localhost:{port}/v1/completions",
                                 json={"model": "Qwen/Qwen2.5-3B-Instruct",
                                       "prompt": prompt, "max_tokens": 100})
            elapsed = time.time() - t
            print(f"{name} ({prompt[:30]}...): {elapsed*1000:.0f} ms")
```

Expected result: VRS-style structured prompts should be 2-3x faster with speculative decoding. Free-form prompts may only improve by 1.2-1.5x.

**Important:** vLLM speculative decoding can conflict with continuous batching. Measure single-request behavior to isolate the speculative decoding effect.

This benchmark supports a concrete VRS operation decision:

- If the verifier processes sequential single requests, enable speculative decoding.
- If the verifier processes batched concurrent requests, disable speculative decoding.

---

## 5.9 Self-Assessment Questions

Answer Q5.1-Q5.5 from the source book, plus the following:

1. **Mathematical guarantee:** Speculative decoding output has exactly the same distribution as the target model output. Why is this decisive for quality? Does the guarantee hold for both greedy decoding and sampling?

2. **VRS verifier fit:** The VRS verifier emits structured JSON output. Give three reasons this helps speculative decoding, and one possible disadvantage.

3. **Memory vs. latency trade-off:** In a VRS scenario with a 16 GB RTX 5080 and a Cosmos-Reason2-2B target, choose among vanilla SD, Medusa, and EAGLE. Which is most reasonable, and why?

4. **Next-generation hardware:** How do B200/RTX 5080 FP4 and larger L2/HBM change the cost-benefit of speculative decoding? Is FP4 + SD a new possible sweet spot?

5. **Why SD does not accelerate prefill:** Speculative decoding only applies to the decode loop. Why does it not speed up prefill? What would be needed theoretically to accelerate prefill?

### Expected Answer Sketch

1. **Why the mathematical guarantee matters:** Speculative decoding does not buy latency by trading away quality. The output distribution is exactly preserved. That means quality benchmarks should not drop after enabling SD. For greedy decoding, T=0, the guarantee is even stronger: the token sequence is exactly the same. For sampling, the distribution is the same, although individual samples can differ. **For the VRS verifier:** enabling SD should not change verdict accuracy or false-positive rate, which removes the largest production adoption concern.

2. **Three reasons VRS is favorable for SD:**

   - **Structured JSON template:** schema tokens such as `"verdict":` and `"confidence":` are nearly deterministic, so α is very high, around 0.95.
   - **Short output:** verdict plus rationale is about 50-100 tokens, so SD overhead is acceptable.
   - **Selective invocation:** the VRS verifier only processes stable candidates, so concurrency is low and the workload is close to the ideal SD regime.

   **Possible disadvantage:** the prompt and output distribution are highly structured, so the draft may behave very similarly to the target most of the time. That is good for acceptance, but if the draft is too small, mismatch can still appear in the semantically important fields. Draft size still needs measurement.

3. **VRS on RTX 5080 16GB:**

   - Cosmos-Reason2-2B BF16 = about 4 GB, or about 1 GB with AWQ-INT4.
   - Vanilla SD with a Qwen2.5-0.5B-Instruct draft adds about 1 GB in BF16. Total is about 5 GB, so memory is available.
   - Medusa adds only about 50 MB of heads. It is the most memory-light, but if α is only around 0.6, the acceleration is smaller.
   - EAGLE-2 uses a small draft, around hundreds of MB, plus EAGLE training. It can reach α above 0.85 and offers the best speed.

   Decision: **try vanilla SD first** because it is production-ready and easy to validate. Treat EAGLE-2 as the longer-term optimization. Medusa is less attractive for this VRS path because sequential reasoning in the rationale can matter.

4. **B200 + FP4 + SD:**

   - Native FP4 speeds up the compute path, so target verification itself becomes faster.
   - Larger L2/HBM means draft and target working sets are more likely to stay resident, especially at small batch.
   - A plausible sweet spot is FP4-quantized draft + FP4-quantized target + SD, potentially 5-7x faster than a BF16 baseline.
   - The unknown is FP4's effect on acceptance rate. Week 4 showed that NF4 increased perplexity by about 8%, so FP4 can also cause distribution drift. Measurement is required.

5. **Why prefill is not accelerated by SD:** Prefill already processes the full prompt in parallel. Speculative decoding breaks the "one token at a time" sequential bottleneck in decode, but prefill does not have that bottleneck. A theoretical workaround would be something like chunked prefill or prompt compression, where a draft first proposes a representation and the target verifies it. That is no longer standard speculative decoding.

---

## 5.10 VRS Reflection

### Why the VRS Verifier Is a Textbook SD Case

Look at the VRS architecture from a speculative decoding perspective:

| VRS property | SD implication |
|---|---|
| Two-stage cascade: fast + slow | The verifier is sequential and latency-sensitive |
| Event-driven verifier invocation | Low concurrency, usually batch=1 single-stream |
| Structured JSON output | Very high α because the template is deterministic |
| Short output: verdict + rationale | 50-100 tokens, easy to amortize SD overhead |
| Verdict quality is business-critical | SD's distribution-preservation guarantee matters |
| Latency-critical alerting | Lower TPOT becomes real business value |

**These properties match the ideal SD use case.** The strongest conclusion from this week is: **for the VRS verifier, not evaluating speculative decoding would leave significant latency reduction on the table.**

### Quantitative Accounting

**Baseline, current VRS with no SD:**

```
Cosmos-Reason2-2B BF16 verifier:
- Prefill: image tokens + watch-policy template, about 500 tokens
- Decode: verdict + confidence + rationale, about 100 tokens

Per-verification latency, RTX 5080, BF16:
- Prefill: about 100 ms, needs measurement
- Decode: 100 tokens x 12 ms = 1.2 s
- Total: about 1.3 s

-> alert TTR, time to result: about 1.3 s after stable candidate
```

**With AWQ-INT4, applying the Week 4 result:**

```
- Prefill: about 50 ms, 2x speedup
- Decode: 100 x 6 ms = 0.6 s, 2.14x from Week 4
- Total: about 0.65 s
```

**With AWQ-INT4 + vanilla SD, assuming α=0.75:**

```
- Prefill: about 50 ms, no SD effect
- Decode: 0.6 s / 2.6, α=0.75 speedup = 0.23 s
- Total: about 0.28 s
```

**With AWQ-INT4 + EAGLE-2 SD, assuming α=0.85:**

```
- Prefill: about 50 ms
- Decode: 0.6 s / 4.7, α=0.85 and K=4 = 0.13 s
- Total: about 0.18 s
```

**Total reduction:**

- BF16 baseline: 1.3 s
- INT4 + EAGLE-2 SD: 0.18 s
- **About 7x speedup**

### VRS System Improvement Priority

1. **Immediate, 1-2 weeks:**
   - Wrap the verifier with vLLM, assuming the current path uses Hugging Face transformers.
   - Apply AWQ-INT4 quantization, using the `tiny.yaml` profile and the Week 4 result.
   - Test vanilla SD with a Qwen2.5-0.5B draft.

2. **Medium-term, 1-2 months:**
   - Measure verifier α with Lab 2.
   - Run an SD on/off A/B test and compare verdict accuracy.
   - Add SD acceptance rate to production logging.

3. **Long-term, 3-6 months:**
   - Train EAGLE-2 based on Cosmos-Reason2-2B.
   - Improve the VRS verifier output schema for SD optimality:
     - continuous confidence -> discrete categorical values, such as `low` / `med` / `high`
     - keep rationale short, ideally one sentence

### Schema Redesign Possibility

The VRS verifier output schema can be redesigned to be more SD-friendly:

```json
// Current
{
  "verdict": "yes",
  "confidence": 0.87,
  "rationale": "Long text..."
}

// SD-optimized
{
  "verdict": "yes",
  "confidence_band": "high",
  "rationale_template": "fall_pose_static",
  "rationale_details": "8s no movement"
}
```

Continuous confidence is harder for the draft to match exactly. A categorical confidence band is easier to accept, clearer for humans, and often sufficient for downstream policy decisions. This is an example where inference optimization knowledge feeds back into product schema design.

---

## 5.11 Deliverables

After this week, keep the following in your notes:

1. **Toy SD implementation and α measurement:** results from Labs 1 and 2.
2. **vLLM SD benchmark:** Lab 3, comparing VRS-style prompts and free-form prompts.
3. **VRS verifier α measurement:** quantitative acceptance rate for structured output.
4. **VRS improvement proposal:** SD adoption and schema-redesign priorities from Section 5.10.

---

## 5.12 Preview of Week 6

Next week covers KV cache optimization and vLLM: the OS virtual memory analogy behind PagedAttention, how prefix caching works in multi-turn dialogue, and StreamingLLM's attention sink. For the VRS multi-stream scenario, we will measure the savings from sharing the same watch-policy template across N streams with prefix caching. We will also cover why vLLM became the de facto standard for production serving, and which parts it solves automatically when integrated into the VRS verifier.
