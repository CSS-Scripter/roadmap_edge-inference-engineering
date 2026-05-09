# Step 3: CPU Optimization Fundamentals

## Context

This is where your C++ background starts to pay off significantly. Most ML practitioners skip this layer entirely. Understanding it is what separates someone who can tune inference from someone who can actually optimize it.

---

## Concepts to Learn

### Memory Hierarchy

- L1/L2/L3 cache sizes and typical latencies (know approximate numbers)
- Cache lines: what they are, why they're 64 bytes, what a cache miss costs
- Temporal vs spatial locality
- Why cache-oblivious algorithms exist

### SIMD

- What SIMD is: one instruction operating on multiple data lanes simultaneously
- AVX2 on x86: 256-bit registers, 8 floats or 16 int16s per instruction
- ARM NEON: 128-bit registers, dominant on mobile/edge ARM chips
- When the compiler auto-vectorizes and when it doesn't
- Intrinsics: writing SIMD explicitly in C++ when the compiler fails you
- How quantized matrix multiplication exploits SIMD (INT8 dot products)

### Threading and Parallelism

- Thread creation overhead — why thread pools exist
- False sharing: when two threads write to different variables on the same cache line
- Work partitioning strategies for matrix operations
- OpenMP basics for simple parallel loops

### Memory Layout for ML Workloads

- Row-major vs column-major storage
- Why matrix multiplication performance depends heavily on memory layout
- Tiling/blocking: restructuring loops so data fits in L1/L2 cache
- Stride access patterns and why they kill performance

### Profiling Tools

- `perf` on Linux: basic usage, cache miss rates, branch misprediction
- Valgrind/cachegrind for cache simulation
- VTune (Intel) or Instruments (Apple) for deeper analysis
- How to read a flamegraph

---

## Resources

|Resource|What It Covers|
|---|---|
|[Computer Systems: A Programmer's Perspective — Ch. 6](https://csapp.cs.cmu.edu/)|Memory hierarchy — this chapter is mandatory reading|
|[Agner Fog — Optimizing C++](https://agner.org/optimize/optimizing_cpp.pdf)|CPU optimization from a practitioner — free PDF|
|[Agner Fog — Instruction Tables](https://agner.org/optimize/instruction_tables.pdf)|Latency/throughput for x86 instructions|
|[Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)|Reference for AVX2 intrinsics|
|[ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)|Reference for NEON|
|[Denis Bakhvalov — Performance Analysis and Tuning on Modern CPUs](https://book.easyperf.net/perf_book)|Free PDF, modern CPU profiling|

---

## Models

Use your ONNX models from Step 1 for the profiling assignment. No new models needed — the focus here is CPU fundamentals, not ML model specifics. ResNet-50 or MobileNetV2 are fine.

---

## Assignments

### Assignment 1: Matrix Multiply, Three Ways

Implement matrix multiplication (512x512 floats) three ways:

1. **Naive triple loop**
2. **Cache-blocked** — tile size derived from your L1 cache size
3. **SIMD** — AVX2 (x86) or NEON (ARM) intrinsics written explicitly

Benchmark all three. Write an explanation of why each is faster than the previous, citing cache line behavior and SIMD lane utilization. Your SIMD version should be at least 4x faster than naive.

### Assignment 2: False Sharing Hunt

Write a program demonstrating false sharing:

- Two threads each incrementing their own counter in a tight loop
- Version A: counters adjacent in memory (same cache line)
- Version B: counters padded to separate cache lines

Measure the difference — it should be dramatic (often 5-10x). Write an explanation at the hardware level.

### Assignment 3: Profile a Real Workload

Run your Step 1 ONNX benchmark under `perf stat` (Linux) or Instruments (macOS). Capture cache miss rates, IPC, and branch misprediction rate. Write a short analysis: is this workload compute-bound or memory-bound, and what would you investigate first?

### Assignment 4: INT8 Dot Product

Implement a dot product of two INT8 vectors using SIMD intrinsics, accumulating into INT32. This is the core operation inside quantized matrix multiplication and understanding it here pays dividends in Steps 4 and 5.

---

## Exit Criteria

Before moving on you must be able to:

- [ ] Explain what a cache line is, why it's 64 bytes, and give a concrete example of code that causes excessive cache misses
- [ ] Explain what SIMD is, how many floats AVX2 processes per instruction, and when the compiler fails to auto-vectorize
- [ ] Explain false sharing and write code that demonstrates it
- [ ] Explain cache blocking and derive an appropriate tile size from L1 cache size
- [ ] Read a basic `perf stat` output and identify whether a workload is compute-bound or memory-bound
- [ ] Explain why INT8 matrix multiply is faster than FP32 beyond just "fewer bytes"

---

## Assessment Prompt

```
I am studying CPU optimization for inference engineering. I have:
- Implemented and benchmarked naive, cache-blocked, and SIMD matrix multiplication
- Demonstrated and measured false sharing
- Profiled a real ONNX model with perf
- Implemented an INT8 SIMD dot product

Please assess me:
1. Ask me to explain cache blocking in matrix multiplication — I should derive tile size
   from first principles given a cache size, not just state a formula
2. Ask me why INT8 SIMD is faster than FP32 beyond just "smaller data" —
   push back if I give a surface answer
3. Ask me to describe what false sharing is and how padding fixes it at the hardware level
4. Give me a perf stat output (make one up with realistic numbers) and ask me to diagnose
   whether the workload is compute-bound or memory-bandwidth-bound
5. Strict grading throughout. Final verdict at the end.
```
