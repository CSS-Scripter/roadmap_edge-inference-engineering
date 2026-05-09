# Step 6: Hardware-Specific Backend (Pick One)

## Context

Edge inference lives or dies on hardware-specific optimization. Each platform has its own acceleration stack, its own constraints, and its own failure modes. You pick one to go deep on — breadth comes later.

---

## Choose Your Target Platform

|Platform|Acceleration Stack|Best If You...|
|---|---|---|
|**Apple Silicon**|Core ML, Metal Performance Shaders|Use macOS, target iPhone/Mac apps|
|**Qualcomm**|QNN, SNPE|Target Android, embedded Snapdragon|
|**Intel**|OpenVINO, oneDNN, Intel NPU|Target Windows PCs, Intel NPU (Meteor Lake+)|
|**NVIDIA Jetson**|TensorRT, cuDNN|Target embedded GPU platforms|

Given your Tauri/desktop background, **Intel OpenVINO or Apple Core ML** are the most immediately applicable.

---

## Concepts to Learn (Common to All Platforms)

- How the platform maps ONNX or PyTorch models to hardware-specific operators
- What operators are natively supported vs require decomposition or CPU fallback
- The NPU/neural engine architecture: how it differs from a CPU and a GPU
- Power and thermal constraints: how they affect sustained inference performance
- Platform profiling tools: measuring which layer runs on which compute unit
- Model format conversion: ONNX → platform native format

---

## Platform-Specific Resources

### Apple Core ML

|Resource|What It Covers|
|---|---|
|[Core ML Documentation](https://developer.apple.com/documentation/coreml)|Official docs|
|[coremltools](https://apple.github.io/coremltools/)|Model conversion from ONNX/PyTorch|
|[WWDC Core ML sessions](https://developer.apple.com/videos/ml-vision/)|Architecture and optimization|

### Intel OpenVINO

|Resource|What It Covers|
|---|---|
|[OpenVINO Documentation](https://docs.openvino.ai/)|Official docs|
|[OpenVINO Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)|ONNX → IR conversion|
|[OpenVINO Benchmarking Tool](https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html)|Performance measurement|
|[Intel NPU documentation](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/npu.html)|Intel NPU specifics|

### Qualcomm QNN

|Resource|What It Covers|
|---|---|
|[Qualcomm AI Hub](https://aihub.qualcomm.com/)|Pretested model performance data|
|[Snapdragon Neural Processing SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)|SNPE SDK|

---

## Models

|Model|Source|Why It's Useful|
|---|---|---|
|**MobileNetV2**|ONNX Model Zoo|Simple, well-tested, supported across all platforms. Good conversion baseline.|
|**Whisper tiny**|HuggingFace via `optimum`|Encoder-decoder with mixed op types — good for finding fallback issues on real models.|
|**Llama 3.2 1B**|`bartowski/Llama-3.2-1B-Instruct-GGUF`|If your platform supports LLM inference, this is the right scale to test with.|

Start with MobileNetV2 for conversion and profiling. Introduce Whisper to stress-test EP fallback. Only attempt LLM inference if your platform has documented support.

---

## Assignments

### Assignment 1: Convert and Run

Take MobileNetV2 from ONNX. Convert it to your platform's native format. Verify output matches the original ONNX output within acceptable tolerance.

Document every conversion issue: unsupported operators, shape problems, dtype mismatches.

### Assignment 2: Layer-by-Layer Profiling

Use your platform's profiling tool to measure per-layer latency. Identify which layers run on NPU vs CPU fallback, why the fallback layers couldn't run on the accelerator, and what you'd need to change to move them.

### Assignment 3: Quantization on Platform

Apply INT8 quantization and re-run. Measure latency difference vs FP32, accuracy difference, and whether the quantized model now fits entirely on the accelerator.

### Assignment 4: Optimize One Bottleneck

Pick the single worst-performing layer from your profiling data. Research and implement an improvement: operator fusion, shape change, alternative quantization, replacing an unsupported op with a supported equivalent. Measure the result.

---

## Exit Criteria

Before moving on you must be able to:

- [ ] Explain how your chosen platform's NPU differs architecturally from a CPU and why certain ops map well to it and others don't
- [ ] Convert a model from ONNX to your platform's format and debug conversion errors without help
- [ ] Interpret your platform's profiler output and identify CPU fallback layers and their causes
- [ ] Explain why quantization is often required (not just beneficial) for NPU acceleration
- [ ] Have end-to-end experience: model in → platform-optimized inference out, with measured performance numbers

---

## Assessment Prompt

```
I am studying hardware-specific inference optimization on [YOUR CHOSEN PLATFORM] as part of
an inference engineering curriculum. I have:
- Converted a model from ONNX to the platform's native format and debugged conversion errors
- Run per-layer profiling and identified NPU vs CPU fallback layers
- Applied INT8 quantization and measured its effect on latency and accuracy
- Optimized one specific bottleneck layer

Please assess me:
1. Ask me to explain how the NPU on my chosen platform differs from a CPU for ML workloads —
   I should get specific about data flow, parallelism, and memory architecture
2. Ask me about a specific conversion error I hit and how I resolved it —
   push back if my explanation is vague
3. Ask me why a specific operator type might not be supported on the NPU
   and what the fallback behavior is
4. Ask me to reason through: if 2 out of 20 layers can't run on the NPU,
   is it worth trying to fix it, and how would I decide?
5. Strict grading. Final verdict.
```
