# Roadmap to Inference Engineering

The goal of this repository is to improve my skills and knowledge towards that of an inference engineer. My focus for now is more edge/on-device inference engineering. 

I had a LLM create a roadmap for me, which later evolved into more concrete tasks, resources and targets.

The roadmap overview:
1. ONNX Runtime In Depth
2. PyTorch Basics + Transformer Architecture
3. CPU Optimization Fundamentals
4. Quantization - Deep and Edge-Focused
5. llama.cpp In Depth
6. Hardware Specific Backend

More in-depth/concrete descriptions for each item will be in the relevant folder's readme file.

Each file contains:

- **Concepts** — what you need to understand
- **Resources** — what to study
- **Models** — specific open source models to use for assignments
- **Assignments** — hands-on work to apply the knowledge
- **Exit Criteria** — specific things you must be able to do/explain before moving on
- **Assessment Prompt** — paste this into Claude to test your readiness

Complete assignments before running the assessment. The assessment is not a quiz — it's a technical conversation where gaps will surface naturally.


---

Here are some directions to move towards after finishing all six steps of the roadmap.

**Deepen on your platform** — contribute to llama.cpp's backend for your chosen hardware, or build something end-to-end that ships.

**Explore MLC-LLM / Apache TVM** — MLC-LLM uses TVM to compile LLMs across backends (Metal, Vulkan, WebGPU, CUDA) from a unified codebase. It's a cross-platform approach that sits above the hardware-specific layer you've just learned. Having the hardware depth from Step 6 makes you effective when MLC-LLM's abstractions break, which they do. Without that depth, you'd be debugging in the dark.

**Bridge to cloud** — your remaining gap is GPU programming. With this foundation, CUDA will be hardware constraint learning rather than starting from zero. The GPU MODE lecture series or NVIDIA's own CUDA course would close it efficiently.

**Build a portfolio** — open source contributions to llama.cpp, a benchmark comparison, or a production tool others use is worth more than any certification.
