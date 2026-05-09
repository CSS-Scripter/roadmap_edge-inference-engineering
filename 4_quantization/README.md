# Step 4: Quantization — Deep and Edge-Focused

## Context

You understand the concept. This step builds the mathematical and practical depth needed to apply, evaluate, and debug quantization — and to reason about whether a model actually fits in a memory budget. On edge hardware these two problems are inseparable.

---

## Concepts to Learn

### Quantization Mathematics

- Affine quantization: `x_q = round(x / scale + zero_point)`, dequantize: `x = (x_q - zero_point) * scale`
- Symmetric vs asymmetric quantization
- Per-tensor vs per-channel vs per-group quantization — accuracy/performance tradeoffs
- Quantization error: clipping vs rounding error, how range affects both
- Why outlier activations (common in LLMs) break naive INT8 quantization

### Quantization Schemes for LLMs

- Post-training quantization (PTQ) vs quantization-aware training (QAT)
- GPTQ: layer-wise PTQ using second-order information — conceptual understanding
- AWQ (Activation-Aware Weight Quantization) — how it handles outlier activations
- KV cache quantization: why it's different from weight quantization

### GGUF and K-Quant Superblock Structure

Basic GGUF types (Q4_0, Q4_1, Q8_0) use a flat block structure: a fixed block of 32 weights shares a single scale factor. Simple, but one outlier weight forces the scale to cover a wide range, wasting precision on everything else.

K-quants (Q4_K, Q5_K, Q6_K) use a two-level hierarchical superblock structure:

- A **superblock** spans 256 weights
- Each superblock is divided into **8 sub-blocks** of 32 weights
- Each sub-block has its own **scale** and **minimum value** stored at reduced precision
- The superblock stores a **super-scale** and **super-min** in FP16 that all sub-block scales are relative to

This hierarchy lets scale factors adapt to local weight distributions without the overhead of per-weight scales. A region with a narrow range gets a tight scale; an adjacent region with a wider range gets its own. The result is meaningfully better accuracy than Q4_0 at essentially the same bit rate.

The suffix letters in names like Q4_K_M indicate a mixed-precision strategy: **M (medium)** uses Q4_K for most layers but Q6_K for sensitive layers (typically first and last). **S (small)** uses Q4_K throughout. This matters because embedding and output layers are disproportionately sensitive to quantization error.

### Memory Budgeting

On edge devices, RAM is the hard constraint everything else bends around. You must be able to calculate total memory requirements before loading a model.

**Components of total inference memory:**

**1. Model weights:** `num_parameters × bytes_per_weight`

- FP32: 4 bytes/param → 7B model = 28 GB
- FP16/BF16: 2 bytes/param → 7B model = 14 GB
- INT8: 1 byte/param → 7B model = 7 GB
- Q4 (~0.5 bytes/param + scale overhead) → 7B model ≈ 3.5–4 GB

**2. KV cache:** `2 × num_layers × num_kv_heads × head_dim × context_length × bytes_per_element`

- The factor of 2 is for K and V
- Grows linearly with context length — this is why long contexts are expensive on edge
- Example: Llama 3.2 1B (16 layers, 8 KV heads, 64 head dim, FP16) at 4096 context ≈ 128 MB

**3. Activations:** bounded by the largest intermediate tensor in a single forward pass. Standard attention's intermediate N×N matrix dominates here; Flash Attention substantially reduces this.

**4. Runtime overhead:** allocator metadata, tokenizer, framework — typically 50–200 MB.

**Practical skill:** given device RAM, model spec, and quantization level, calculate the maximum context length before running out of memory. You should be able to do this on paper.

KV cache quantization is an additional lever when context length is the binding constraint rather than weight size.

---

## Resources

|Resource|What It Covers|
|---|---|
|[Hugging Face — Quantization Conceptual Guide](https://huggingface.co/docs/transformers/quantization)|Good practical overview|
|[GPTQ Paper](https://arxiv.org/abs/2210.17323)|Conceptual understanding|
|[AWQ Paper](https://arxiv.org/abs/2306.00978)|Activation-aware quantization|
|[LLM.int8() Paper](https://arxiv.org/abs/2208.07339)|Explains the outlier activation problem clearly|
|[GGUF format specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)|GGUF binary format spec|
|[llama.cpp quantization README](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)|Practical quantization with llama.cpp|
|[ggml-quants.c source](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c)|The actual K-quant implementation — read alongside the concepts above|
|[Tim Dettmers — Which GPU for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)|Practical perspective on precision tradeoffs|

---

## Models

|Model|Source|Why It's Useful|
|---|---|---|
|**Llama 3.2 1B Instruct**|`bartowski/Llama-3.2-1B-Instruct-GGUF` on HuggingFace|Smallest capable modern LLM. Good perplexity baseline. Use for memory budgeting exercises where you want realistic numbers.|
|**Qwen2.5 0.5B Instruct**|`Qwen/Qwen2.5-0.5B-Instruct-GGUF` on HuggingFace|Very small, fast iteration for quantization comparison experiments.|
|**SmolLM2 135M**|`HuggingFaceTB/SmolLM2-135M-GGUF` on HuggingFace|So small that quantization artifacts are immediately visible — educational for error analysis.|

Use **Qwen2.5 0.5B or SmolLM2** for the quantization comparison assignment. Use **Llama 3.2 1B** for the memory budgeting assignment.

---

## Assignments

### Assignment 1: Implement Affine Quantization

From scratch (no libraries), implement:

- `quantize(tensor, bits, symmetric=True/False)` — returns quantized values, scale, zero_point
- `dequantize(quantized, scale, zero_point)` — reconstructs float tensor
- Measure MSE quantization error at INT8, INT4, INT2 for a random normal tensor
- Plot error vs bit width. Note the symmetric vs asymmetric difference for non-zero-centered distributions.

### Assignment 2: Quantize a Real LLM

Download Qwen2.5 0.5B in F16 GGUF. Using llama.cpp's `quantize` tool, produce: Q8_0, Q5_0, Q4_0, Q4_K_S, Q4_K_M.

For each measure: file size, inference latency (`llama-bench`), perplexity (`llama-perplexity`).

Build a table. Write a focused analysis on Q4_K_M vs Q4_0 specifically — the accuracy difference should be visible despite similar size, and you should explain it using the superblock structure.

### Assignment 3: Inspect a GGUF File

Write a script that opens a GGUF file and prints:

- All metadata fields
- Every tensor: name, shape, quantization type, size in bytes

Read the GGUF spec to do this — don't guess the format.

### Assignment 4: Memory Budget Calculator

Build a script that takes as input:

- Number of parameters, quantization level
- Number of layers, KV heads, head dimension
- Target context length, available RAM

And outputs: weight memory, KV cache memory, estimated total, and maximum context length that fits.

Validate it against reality: load Llama 3.2 1B at different context lengths and compare your predictions to actual memory usage.

### Assignment 5: Find the Outliers

Load a small transformer in PyTorch. Record activation tensors at each layer. For each, compute range and percentage of outliers (>3 standard deviations). Write an explanation of why outliers break naive INT8 and how AWQ approaches the problem.

---

## Exit Criteria

Before moving on you must be able to:

- [ ] Derive scale and zero_point for affine quantization from a tensor's min/max
- [ ] Explain why per-channel quantization is more accurate than per-tensor and what the performance cost is
- [ ] Explain what outlier activations are and what LLM.int8() does about them
- [ ] Explain the K-quant superblock structure and why Q4_K_M is more accurate than Q4_0 at similar size
- [ ] Given a model spec and RAM budget, calculate maximum context length before running out of memory
- [ ] Explain why KV cache memory grows with context length and state the formula
- [ ] Explain why quantizing the KV cache is different from quantizing weights

---

## Assessment Prompt

```
I am studying quantization and memory budgeting for edge LLM inference. I have:
- Implemented affine quantization from scratch and measured error vs bit width
- Quantized a real LLM at multiple precision levels including K-quant variants
  and measured latency/perplexity tradeoffs
- Parsed a GGUF file and read its tensor metadata
- Built a memory budget calculator and validated it against real model memory usage
- Analyzed outlier activations in a transformer model

Please assess me:
1. Ask me to explain affine quantization mathematically and derive scale/zero_point
   for a specific example you give me
2. Ask me to explain the K-quant superblock structure and why Q4_K_M outperforms Q4_0
   at similar bit rates — I should explain the two-level hierarchy specifically
3. Ask me why outlier activations break naive INT8 — push back if I don't reach
   the specific arithmetic failure mode
4. Give me a scenario: 4GB RAM, Llama 3.2 1B (16 layers, 8 KV heads, 64 head dim).
   Ask me to calculate what context length fits at Q4_K_M. I should walk through it explicitly.
5. Ask me to explain the difference between quantizing weights vs quantizing the KV cache
6. Strict grading. Final verdict.
```
