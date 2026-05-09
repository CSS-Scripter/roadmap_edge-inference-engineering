# Step 2: PyTorch Basics + Transformer Architecture

## Context

You don't need to become an ML researcher. The goal is to understand what you're running well enough to reason about its performance characteristics, inspect models, and understand why certain architectural choices create inference challenges.

---

## Concepts to Learn

### PyTorch Fundamentals (Inference-Focused)

- Tensors: shapes, dtypes, device placement (CPU vs CUDA)
- Loading pretrained models (`torch.load`, HuggingFace `from_pretrained`)
- Running inference: `model.eval()`, `torch.no_grad()`, why these matter

**torch.export** — the current standard for capturing a model as a portable, deployable computation graph. Produces an `ExportedProgram` with strict shape and operator guarantees, suitable for ahead-of-time compilation and deployment. This is the replacement for the legacy TorchScript approach.

**torch.compile** — wraps a model to JIT-compile it using Triton/inductor kernels at runtime. This is a performance optimization tool for training or inference, not an export format. The distinction matters: `torch.export` produces a portable artifact you hand to a runtime; `torch.compile` speeds up execution in the same Python process.

**ONNX export** — `torch.onnx.export` still works and in newer PyTorch versions increasingly delegates to `torch.export` internally. You'll use this to bridge to ONNX Runtime.

_Note: TorchScript (`torch.jit.script/trace`) is legacy. You will encounter it in older codebases. Know what it is for recognition but don't invest in learning it deeply._

### Transformer Architecture (Deep)

**Tokenization**

- Byte-pair encoding (BPE) — how text becomes token IDs
- Vocabulary size and its effect on the embedding matrix size

**Embedding Layer**

- Token embeddings vs positional embeddings
- Why positional encodings are needed at all (attention has no inherent position sense)

**Transformer Block**

- Layer normalization: where it sits and why (pre-norm vs post-norm)
- Multi-head self-attention:
    - Q, K, V projections and their shapes
    - Scaled dot-product attention math
    - Why multiple heads
    - Causal masking in decoder-only models and why it's needed
- Feed-forward network (MLP): typical 4x expansion ratio, activation functions (ReLU, SiLU, GELU)
- Residual connections: where they are and why they matter

**Model Variants**

- Encoder-only (BERT) vs decoder-only (GPT/Llama) vs encoder-decoder (T5)
- Why decoder-only is dominant for LLMs
- How layers, heads, and hidden dimension affect compute and memory

**Attention + KV Cache**

- Why KV cache only applies to K and V, not Q
- Multi-head attention vs grouped query attention (GQA) vs multi-query attention (MQA)
- Why GQA/MQA exist: KV cache memory reduction

**Flash Attention — Conceptual Introduction**

Standard attention materializes the full N×N attention score matrix in memory. For a sequence of length N, this is O(n²) memory. At long context lengths this becomes the dominant memory cost and a serious bandwidth bottleneck — the matrix is written to HBM, read back for softmax, written again, read again for the weighted sum.

Flash Attention (Dao et al., 2022) avoids this by tiling the computation: it processes attention in blocks that fit in fast SRAM, computing softmax incrementally using an online algorithm, and never writing the full N×N matrix to slow HBM at all. Result: O(n) memory instead of O(n²), with significant throughput gains from reduced memory bandwidth.

You do not need to understand the tiling algorithm here — that comes in Step 5. What you need now:

- Why standard attention is memory-bound at long sequence lengths
- What Flash Attention solves conceptually (IO-awareness: minimizing slow memory reads/writes)
- That it is the standard attention implementation in virtually every modern inference engine

---

## Resources

|Resource|What It Covers|
|---|---|
|[Karpathy — Zero to Hero: GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)|Build a GPT character-level model from scratch — do not skip this|
|[The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)|Attention is All You Need with line-by-line code annotations|
|[Attention is All You Need (paper)](https://arxiv.org/abs/1706.03762)|Read after the annotated version|
|[Flash Attention paper — intro only](https://arxiv.org/abs/2205.14135)|Read the first 3 pages for conceptual understanding. Full algorithm detail in Step 5.|
|[Llama 2 paper](https://arxiv.org/abs/2307.09288)|Modern decoder-only architecture — GQA, RoPE, RMSNorm|
|[PyTorch Docs — torch.export](https://pytorch.org/docs/stable/export.html)|Current export API|
|[PyTorch Docs — torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html)|Runtime compilation|
|[PyTorch Docs — torch.onnx.export](https://pytorch.org/docs/stable/onnx.html)|ONNX export|
|[HuggingFace Transformers — Quickstart](https://huggingface.co/docs/transformers/quickstart)|Loading and running real models|

---

## Models

|Model|HuggingFace ID|Why It's Useful|
|---|---|---|
|**GPT-2 small**|`gpt2`|Canonical learning model. 117M params, fits anywhere. Karpathy's tutorial reimplements this architecture.|
|**BERT base uncased**|`bert-base-uncased`|Encoder-only. Good architectural contrast to GPT-2.|
|**T5 small**|`t5-small`|Encoder-decoder. Completes the three-architecture comparison.|

Export to ONNX via HuggingFace Optimum:

```bash
pip install optimum
optimum-cli export onnx --model gpt2 gpt2_onnx/
optimum-cli export onnx --model bert-base-uncased bert_onnx/
optimum-cli export onnx --model t5-small t5_onnx/
```

---

## Assignments

### Assignment 1: Build a Tiny GPT

Follow Karpathy's GPT from scratch video and implement a character-level language model. Type the code yourself — don't copy-paste.

Extend it:

- Add a function that prints the shape of every tensor at each stage of a single forward pass
- Add a comment on every major operation explaining what it's doing to the data shape

### Assignment 2: Export and Inspect

Take your tiny GPT or `gpt2` from HuggingFace:

- Export it to ONNX using `torch.onnx.export`
- Also capture it using `torch.export.export()` and inspect the resulting `ExportedProgram`
- Open the ONNX graph with your Step 1 inspector
- Map every ONNX node back to the PyTorch module it came from
- Identify where the attention mechanism lives in the ONNX graph

### Assignment 3: Architecture Comparison

Load GPT-2, BERT base, and T5 small from HuggingFace. For each:

- Count parameters (total and per layer)
- Measure inference latency for a fixed input length
- Explain in writing why their latency profiles differ

### Assignment 4: Sequence Length vs Memory

Using GPT-2, run inference at sequence lengths 128, 256, 512, and 1024 tokens. Measure peak memory at each length. Plot it. Write an explanation connecting the memory growth curve to the O(n²) attention matrix and what Flash Attention changes about this.

---

## Exit Criteria

Before moving on you must be able to:

- [ ] Explain what happens to input data at every stage of a transformer forward pass including tensor shapes
- [ ] Explain why causal masking is needed in decoder-only models and what breaks without it
- [ ] Explain the difference between MHA, GQA, and MQA and why GQA reduces KV cache memory
- [ ] Export a PyTorch model to ONNX and identify the attention operators in the resulting graph
- [ ] Explain what `model.eval()` and `torch.no_grad()` do and why skipping them in inference is wrong
- [ ] Explain why positional encodings are needed and how RoPE differs from learned positional embeddings
- [ ] Explain the difference between `torch.export` and `torch.compile` and when you'd use each
- [ ] Explain why standard attention has O(n²) memory complexity and what Flash Attention addresses conceptually

---

## Assessment Prompt

```
I am studying transformer architecture and PyTorch fundamentals as part of a self-directed
inference engineering curriculum. I have:
- Implemented a GPT character-level model from scratch following Karpathy's tutorial
- Exported models using both torch.onnx.export and torch.export, and compared them
- Loaded and compared BERT, GPT-2, and T5 architecturally
- Measured memory usage vs sequence length and connected it to attention complexity

Please assess my readiness by:
1. Asking me to walk through a transformer forward pass step by step including tensor shapes.
   Interrupt and correct me if I get shapes wrong or skip something important.
2. Asking me to explain the difference between torch.export and torch.compile —
   what each produces, when you'd use each, and what's wrong with TorchScript.
3. Asking me to explain the difference between MHA, GQA, and MQA and why they exist.
4. Asking me to explain why standard attention has O(n²) memory complexity and what
   Flash Attention changes — I should get to SRAM vs HBM without being prompted.
5. Asking me one question about what breaks (and why) if you run inference without model.eval().
6. Grading each answer strictly: correct, partially correct, or wrong with explanation.
7. Final verdict: ready to continue, or what to revisit.

Do not soften incorrect answers.
```
