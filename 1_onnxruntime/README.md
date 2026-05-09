# Step 1: ONNX Runtime In Depth

## Context

You already use ONNX Runtime at an integration level. This step closes the gap between "I load and run models" and "I understand what the runtime is doing and can control it." This is low-hanging fruit given your existing codebase.

---

## Concepts to Learn

### Graph Internals

- ONNX model structure: graph, nodes, inputs/outputs, initializers
- Operator types and their attributes
- Shape inference and dynamic vs static shapes
- How to detect unsupported operators for a given execution provider

### SessionOptions

- Graph optimization levels (DISABLE_ALL, ENABLE_BASIC, ENABLE_EXTENDED, ENABLE_ALL) — what each level does and the tradeoffs
- Thread pool configuration: intra-op vs inter-op parallelism
- Memory arena allocators and when they matter
- Execution mode (sequential vs parallel)

### Execution Providers

- How ONNX Runtime decides which EP runs which node
- EP capability queries — how a node gets assigned to CPU fallback
- How to inspect which nodes ran on which EP after a session runs
- Common reasons for CPU fallback (unsupported op, unsupported datatype, shape constraints)

### Profiling

- Enabling and reading the ONNX Runtime profiler output (JSON)
- Identifying operator-level bottlenecks
- Understanding what "kernel time" vs "fence time" means in profiler output

---

## Resources

|Resource|What It Covers|
|---|---|
|[ONNX Runtime Docs — Session Options](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)|Threading, optimization levels|
|[ONNX Runtime Docs — Execution Providers](https://onnxruntime.ai/docs/execution-providers/)|EP configuration and fallback behavior|
|[ONNX Runtime Docs — Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance/)|Practical tuning guide|
|[ONNX Python API](https://onnx.ai/onnx/api/)|Graph inspection with the `onnx` library (separate from `onnxruntime`)|
|[Netron](https://netron.app/)|Visual graph inspector — use alongside code inspection|
|[ONNX Runtime Profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)|How to enable and read profiler output|

**Note:** The `onnx` Python library (for graph inspection) and `onnxruntime` (for execution) are separate packages. You need both.

---

## Models

|Model|Source|Why It's Useful|
|---|---|---|
|**ResNet-50**|[ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet)|Simple, well-documented graph. Good starting point for inspection.|
|**MobileNetV2**|[ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)|Edge-relevant depthwise convolutions behave differently from standard convolutions — good for EP fallback investigation.|
|**YOLOv8n**|`ultralytics` Python package — `model.export(format='onnx')`|More complex graph, good for finding EP fallback issues.|
|**Whisper tiny**|HuggingFace via `optimum-cli export onnx --model openai/whisper-tiny whisper_onnx/`|Encoder-decoder structure adds graph complexity.|

**Start with ResNet-50 or MobileNetV2.** Complex enough to be interesting, simple enough that you won't get lost.

---

## Assignments

### Assignment 1: Graph Inspector

Take any model from the table above. Write a script (Python or C++) that:

- Prints every node: op type, input names, output names, attributes
- Lists every initializer (weight tensor) with its shape and dtype
- Identifies which nodes would likely fall back to CPU if you used a non-CPU EP

Deliverable: a script and a written summary of what you found in your model.

### Assignment 2: SessionOptions Benchmark

Write a benchmarking harness that runs inference 100 times and measures average latency. Vary:

- Intra-op thread count (1, 2, 4, 8, max)
- Graph optimization level (all four levels)
- Execution mode (sequential vs parallel)

Table the results. Write a short explanation of which settings performed best and why.

### Assignment 3: Profiler Analysis

Enable the ONNX Runtime profiler on a model run. From the JSON output, identify:

- The top 3 operators by total execution time
- Whether those operators ran on CPU or another EP
- Any operators that surprised you

Write a short paragraph explaining what you'd investigate next to make this model faster.

---

## Exit Criteria

Before moving on you must be able to:

- [ ] Explain the four graph optimization levels and give a concrete example of what ENABLE_EXTENDED does that ENABLE_BASIC does not
- [ ] Explain the difference between intra-op and inter-op parallelism and when each matters
- [ ] Given a profiler JSON output, identify the bottleneck operator and explain what EP it ran on
- [ ] Explain three concrete reasons a node might fall back to CPU even when a GPU EP is configured
- [ ] Inspect an ONNX graph programmatically and describe its structure without using Netron

---

## Assessment Prompt

```
I am studying ONNX Runtime internals as part of a self-directed inference engineering curriculum.
I have completed the following:
- Read the ONNX Runtime documentation on SessionOptions, execution providers, and profiling
- Written a graph inspection script for a real model
- Run a benchmarking harness comparing thread counts and optimization levels
- Analyzed profiler output from a real inference session

Please assess my readiness to move on by doing the following:
1. Ask me 4-5 technical questions covering: graph optimization levels, EP fallback behavior,
   threading model, and profiler output interpretation
2. After each answer, tell me directly if I am correct, partially correct, or wrong,
   and explain what the right answer is
3. If I use vague or hand-wavy language, push back and ask me to be more specific
4. At the end, give me a verdict: ready to move on, or specific topics I need to revisit
5. Do not validate incorrect answers to make me feel better — I want accurate feedback

Start with the first question.
```
