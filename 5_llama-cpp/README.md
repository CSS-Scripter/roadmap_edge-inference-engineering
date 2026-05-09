# Step 5: llama.cpp In Depth

## Context

This is where everything comes together. llama.cpp is a production inference engine written in C that you can read, understand, and contribute to. By the end of this step you should understand it well enough to make meaningful modifications.

---

## Concepts to Learn

### Codebase Architecture

- The relationship between `llama.cpp` (model/runtime logic) and `ggml` (the tensor library)
- How the build system works and how backends are selected at compile time
- The main data structures: `llama_model`, `llama_context`, `llama_kv_cache`
- How a forward pass is constructed as a computation graph in ggml

### ggml Tensor Library

- How ggml represents tensors (no automatic differentiation, forward-only)
- How ggml builds a computation graph lazily and executes it
- Memory management: ggml memory arenas, scratch buffers
- How ggml backends abstract over CPU/Metal/CUDA/Vulkan execution

### Flash Attention — Full Algorithm

Now that you have the conceptual foundation from Step 2, understand the implementation in detail:

**IO-aware analysis:** HBM (high-bandwidth memory, i.e. main device RAM) has high bandwidth but high latency relative to SRAM (on-chip cache). Standard attention makes multiple round trips through HBM: read Q/K/V, write N×N scores, read them for softmax, write softmax output, read it for weighted sum. Flash Attention eliminates all intermediate HBM writes.

**Tiling:** Q, K, V matrices are split into blocks sized to fit in SRAM. For each Q block, all K and V blocks are iterated over. Partial attention results are computed and accumulated in SRAM without ever materializing the full N×N matrix.

**Online softmax:** Standard softmax requires knowing the row maximum before computing any exponentials (for numerical stability). Flash Attention uses a running max and running normalizer that are updated as each K block is processed, enabling correct softmax within the tiling loop without a separate pass.

**Result:** no N×N matrix is ever written to HBM. Attention scores between a Q block and K block are computed, used immediately to weight the V block contribution, and discarded. Total HBM accesses scale as O(n) instead of O(n²).

- Where Flash Attention is implemented in llama.cpp
- Under what conditions llama.cpp falls back to standard attention
- Flash Attention 2 improvements: better work partitioning across warps

### Attention Implementation

- Where multi-head attention is computed and how the KV cache is updated
- How causal masking is applied
- How RoPE is applied in the codebase
- How llama.cpp handles GQA

### Quantized Matrix Multiply

- Where `ggml_mul_mat` is defined and how it dispatches to quantized kernels
- How Q4_K matrix multiplication works at the kernel level
- How AVX2/NEON SIMD is used in the quantized kernels (connects back to Step 3)

### Inference Loop

- Prefill vs decode execution paths
- How batching is handled
- How sampling (temperature, top-p, top-k) works at the code level

### Speculative Decoding

Standard autoregressive generation is sequential — each token depends on the previous. Speculative decoding exploits the fact that verification is parallelizable even when generation is not.

**Naive (draft model):** A small fast draft model generates k candidate tokens autoregressively. The large target model verifies all k in a single parallel forward pass — equivalent to a prefill step, so it's fast. Accepted tokens are kept; the first rejected token triggers a rollback. When the draft model is right, you get multiple tokens per target model call. Requires two models in memory — primarily practical for server deployment.

**Medusa:** Adds multiple prediction heads directly to the base model's final layer. Head 1 predicts t+1, head 2 predicts t+2, and so on. A tree attention mechanism verifies candidate sequences in parallel within the same model call. Memory overhead is minimal — just the extra heads, not a second model. Well-suited to edge deployment.

**EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency):** A lightweight autoregressive draft model that takes the base model's hidden states as input rather than operating from output logits alone. Because it has access to richer internal representations, EAGLE's draft can be much smaller than a standalone model while achieving high acceptance rates. Requires a small second model but dramatically cheaper than naive speculative decoding.

**Early exit:** Monitors intermediate layer outputs during generation. For "easy" tokens where the model is highly confident by layer N, remaining layers are skipped. One model, no extra memory, per-token adaptive compute. Tricky to implement without accuracy regression on harder tokens.

**Practical guidance for edge:**

- Tight memory → Medusa or early exit
- Some memory headroom + highest throughput → EAGLE
- Naive draft model → server/cloud deployment

**Break-even analysis:** speculative decoding only wins if the acceptance rate is high enough to offset the overhead of running the draft. Understanding the break-even point is a practical skill — measure it on your hardware.

---

## Resources

|Resource|What It Covers|
|---|---|
|[llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)|Primary source — read the code|
|[ggml GitHub](https://github.com/ggerganov/ggml)|The underlying tensor library|
|[llama.cpp — ARCHITECTURE.md](https://github.com/ggerganov/llama.cpp/blob/master/docs/development/ARCHITECTURE.md)|Start here before reading code|
|[Flash Attention paper (full)](https://arxiv.org/abs/2205.14135)|Read in full now — you have the context|
|[Flash Attention 2 paper](https://arxiv.org/abs/2307.08691)|Improved work partitioning|
|[Medusa paper](https://arxiv.org/abs/2401.10774)|Multiple decoding heads|
|[EAGLE paper](https://arxiv.org/abs/2401.15077)|Hidden-state-based draft model|
|[llama.cpp speculative decoding docs](https://github.com/ggerganov/llama.cpp/blob/master/examples/speculative/README.md)|How it's implemented in llama.cpp|
|[Georgi Gerganov's blog](https://ggerganov.com/)|Author's own explanations|

---

## Models

|Model|Source|Why It's Useful|
|---|---|---|
|**Llama 3.2 1B Instruct**|`bartowski/Llama-3.2-1B-Instruct-GGUF`|Primary target model for most assignments|
|**Qwen2.5 0.5B Instruct**|`Qwen/Qwen2.5-0.5B-Instruct-GGUF`|Draft model for speculative decoding experiments — small enough to pair with the 1B|
|**SmolLM2 135M**|`HuggingFaceTB/SmolLM2-135M-GGUF`|Alternative draft model — extreme size difference makes acceptance rate dynamics clear|

---

## Assignments

### Assignment 1: Build and Instrument

Build llama.cpp from source with debug symbols. Run inference on Llama 3.2 1B. Step through a single forward pass with a debugger.

Write a walkthrough: what function is called first, where the computation graph is built, where the KV cache is updated, where sampling happens.

### Assignment 2: Trace a Single Attention Head

Find where multi-head attention is computed. Add temporary logging that prints the shape and a few values of Q, K, V, and the output for a single head on a single layer during a single generation step.

Find where Flash Attention is used in llama.cpp. Identify the conditions under which it activates vs falls back to standard attention.

### Assignment 3: Benchmark the Backends

Compile with CPU-only, then with your available accelerated backend (Metal, CUDA, or Vulkan). Benchmark the same model on each. Measure prefill tokens/second and decode tokens/second **separately**.

Write an analysis of why prefill and decode scale differently, connecting it to the compute-bound vs memory-bound distinction from Step 3.

### Assignment 4: Speculative Decoding Experiment

Run Llama 3.2 1B as the target with SmolLM2 135M or Qwen2.5 0.5B as the draft. Measure:

- Tokens/second with and without speculative decoding
- Token acceptance rate
- Memory usage with both models loaded

Calculate the break-even acceptance rate for your hardware — at what acceptance rate does speculative decoding become worthwhile? Write an analysis of whether it makes sense for your deployment target.

### Assignment 5: Make a Meaningful Modification

Pick one of:

- Add per-layer timing instrumentation to the forward pass
- Improve documentation for a subsystem you now understand well
- Fix an open issue tagged `good first issue`
- Add a sampling strategy not yet implemented

The goal is a change that requires understanding the codebase. Ideally open a PR.

---

## Exit Criteria

Before moving on you must be able to:

- [ ] Explain the relationship between llama.cpp and ggml and why they are separate
- [ ] Trace the execution path from "user provides a token" to "logit is produced"
- [ ] Explain how ggml builds and executes a computation graph
- [ ] Explain how the KV cache is stored and updated (data structure level)
- [ ] Explain Flash Attention's tiling algorithm and online softmax at the algorithmic level
- [ ] Explain the difference between Medusa, EAGLE, and early exit — specifically which require a second model and which don't, and the memory implications of each
- [ ] Calculate the break-even acceptance rate for speculative decoding on a given hardware budget
- [ ] Have made at least one code change to the codebase and verified it behaves correctly

---

## Assessment Prompt

```
I am studying llama.cpp internals as part of an inference engineering curriculum. I have:
- Built llama.cpp from source and stepped through a forward pass with a debugger
- Traced Q, K, V tensor shapes in a specific attention head
- Located Flash Attention in the codebase and identified when it activates
- Benchmarked CPU vs accelerated backends measuring prefill and decode separately
- Run speculative decoding experiments and measured acceptance rate and throughput
- Made a code modification to the codebase

Please assess me:
1. Ask me to explain the relationship between llama.cpp and ggml —
   specifically what responsibility each owns
2. Ask me to explain Flash Attention's tiling algorithm in detail — I should explain
   online softmax, why the N×N matrix is never materialized, and the SRAM vs HBM
   distinction. Push back if I wave my hands.
3. Ask me to explain the difference between Medusa and EAGLE — which requires a second model,
   what each uses as its draft signal, and when you'd choose one over the other on an edge device
4. Ask me to calculate the break-even acceptance rate for speculative decoding given
   a scenario you describe (e.g. draft model runs at 5x the speed of the target model)
5. Ask me why prefill and decode scale differently across backends
6. Ask me to describe the modification I made and explain why it required
   understanding the codebase
7. Strict grading. Final verdict.
```
