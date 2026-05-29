>
> Write a benchmarking harness that runs inference 100 times and measures average latency. Vary:
> 
> - Intra-op thread count (1, 2, 4, 8, max)
> - Graph optimization level (all four levels)
> - Execution mode (sequential vs parallel)
> 
> Table the results. Write a short explanation of which settings performed best and why.
>

# Research
Let's see what we're actually testing here. We're going to compare intra thread counts, graph optimization levels and execution modes. Let's see what they mean.
## Thread counts
The first thing we need to know is about threading. ONNX recognizes two types of threads that we can configure: intra and inter. Inter threads are for running multiple operators in parallel. Intra threads are used within a single operator. The default inter thread count is equal to the amount of logical cores on the machine, which in my case is 16.

## Graph Optimization
Graph optimization happens cumulatively across three stages:
1. Basic: pre-calculate static nodes, remove redundant nodes and apply simple merges, like moving an Add node, that directly follows a Conv node, into that Conv node's bias.
2. Extended: more complex node fusions, which should mainly affect the CPU, CUDA and ROCm execution providers.
3. Layout: change NCHW layout into NCHWc, which is better aligned to the CPU for SIMD operations.
As mentioned above, these happen cumulatively, and for this there are four different optimization levels within ONNXRuntime:
- DISABLE_ALL: None
- ENABLE_BASIC: Basic
- ENABLE_EXTENDED: Basic + Extended
- ENABLE_ALL: Basic + Extended + Layout

## Execution Modes
Execution modes describe how and in what order ONNX should run nodes. There are only two execution modes within ONNX: Sequential and Parallel. Sequential simply means 1 by 1, in the order that they're needed. Parallel will try and find nodes it can run in parallel. The requirement for a node that can be ran being that all inputs are available. Example of when this can happen are when the output of 1 node branches into multiple other nodes.

When running Sequential execution mode, the inter thread count is redundant, as it only runs 1 node at a time.

# Script

```python
import json
import os
import time

import psutil
import onnxruntime as ort
import numpy as np

from tabulate import tabulate


models_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../models"))

modelpath_mobilenet = os.path.join(models_path, "mobilenetv2-7.onnx")
modelpath_yolo      = os.path.join(models_path, "yolov8n.onnx")
modelpath_resnet    = os.path.join(models_path, "resnet50-v1-7.onnx")


def get_intra_thread_options():
    """get options for num_intra_threads, without burning out the CPU"""
    cores = psutil.cpu_count(logical=True)
    if cores is None:
        print("failed to fetch core count")
        return []
    print(f"found {cores} cores")

    intra_thread_opts = []
    threads = 1
    while threads <= cores:
        intra_thread_opts.append(threads)
        threads *= 2

    return intra_thread_opts


def run_benchmark(
        model: str,
        intra_threads: int,
        execution_mode: ort.ExecutionMode,
        optimization_level: ort.GraphOptimizationLevel,
) -> list[float]:
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = intra_threads
    session_options.execution_mode = execution_mode
    session_options.graph_optimization_level = optimization_level

    session = ort.InferenceSession(model, sess_options=session_options)
    image = np.random.randn(*session.get_inputs()[0].shape).astype(np.float32)
    times = []

    session.run(None, {"images": image})

    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {"images": image})
        end = time.perf_counter()
        times.append((end - start) * 1000)       

    return times

def main():
	intra_thread_opts = get_intra_thread_options()
	graph_optimization_levels = [
		ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
		ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
		ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
		ort.GraphOptimizationLevel.ORT_ENABLE_ALL
	]
	execution_modes = [
		ort.ExecutionMode.ORT_PARALLEL,
		ort.ExecutionMode.ORT_SEQUENTIAL
	]
	
	results = []
	for intra_threads in intra_thread_opts:
		for graph_optimization in graph_optimization_levels:
			for execution_mode in execution_modes:
				print(f"running scenario (threads={intra_threads}, optimization={graph_optimization}, exec_mode={execution_mode})")
				result = run_benchmark(
					model=modelpath_yolo,
					intra_threads=intra_threads,
					execution_mode=execution_mode,
					optimization_level=graph_optimization
				)
				results.append({
					"intra": intra_threads,
					"exec": str(execution_mode),
					"optim": str(graph_optimization),
					"results": result
				})
	
	with open("output.json", "w", encoding="utf-8") as f:
		json.dump(results, f, indent=4, ensure_ascii=False)
	
	
	table_headers = ["intra", "exec", "optim", "avg ms"]
	data = []
	for result in results:
		times = result["results"]
		avg_ms = sum(times) / len(times)
		data.append([result["intra"], result["exec"], result["optim"], f"{avg_ms:.2f} ms"])
	
	print(tabulate(data, headers=table_headers, tablefmt="fancy_grid"))


if __name__ == "__main__":
	main()
```
# Results

| intra | exec           | optim               | avg ms    |
| ----- | -------------- | ------------------- | --------- |
| 1     | ORT_PARALLEL   | ORT_DISABLE_ALL     | 111.36 ms |
| 1     | ORT_SEQUENTIAL | ORT_DISABLE_ALL     | 111.16 ms |
| 1     | ORT_PARALLEL   | ORT_ENABLE_BASIC    | 111.56 ms |
| 1     | ORT_SEQUENTIAL | ORT_ENABLE_BASIC    | 110.80 ms |
| 1     | ORT_PARALLEL   | ORT_ENABLE_EXTENDED | 110.50 ms |
| 1     | ORT_SEQUENTIAL | ORT_ENABLE_EXTENDED | 108.94 ms |
| 1     | ORT_PARALLEL   | ORT_ENABLE_ALL      | 90.87 ms  |
| 1     | ORT_SEQUENTIAL | ORT_ENABLE_ALL      | 90.62 ms  |
| 2     | ORT_PARALLEL   | ORT_DISABLE_ALL     | 69.68 ms  |
| 2     | ORT_SEQUENTIAL | ORT_DISABLE_ALL     | 72.11 ms  |
| 2     | ORT_PARALLEL   | ORT_ENABLE_BASIC    | 68.80 ms  |
| 2     | ORT_SEQUENTIAL | ORT_ENABLE_BASIC    | 70.78 ms  |
| 2     | ORT_PARALLEL   | ORT_ENABLE_EXTENDED | 66.49 ms  |
| 2     | ORT_SEQUENTIAL | ORT_ENABLE_EXTENDED | 68.22 ms  |
| 2     | ORT_PARALLEL   | ORT_ENABLE_ALL      | 55.84 ms  |
| 2     | ORT_SEQUENTIAL | ORT_ENABLE_ALL      | 56.19 ms  |
| 4     | ORT_PARALLEL   | ORT_DISABLE_ALL     | 50.00 ms  |
| 4     | ORT_SEQUENTIAL | ORT_DISABLE_ALL     | 45.71 ms  |
| 4     | ORT_PARALLEL   | ORT_ENABLE_BASIC    | 48.07 ms  |
| 4     | ORT_SEQUENTIAL | ORT_ENABLE_BASIC    | 46.04 ms  |
| 4     | ORT_PARALLEL   | ORT_ENABLE_EXTENDED | 45.85 ms  |
| 4     | ORT_SEQUENTIAL | ORT_ENABLE_EXTENDED | 42.50 ms  |
| 4     | ORT_PARALLEL   | ORT_ENABLE_ALL      | 39.47 ms  |
| 4     | ORT_SEQUENTIAL | ORT_ENABLE_ALL      | 36.25 ms  |
| 8     | ORT_PARALLEL   | ORT_DISABLE_ALL     | 50.13 ms  |
| 8     | ORT_SEQUENTIAL | ORT_DISABLE_ALL     | 37.93 ms  |
| 8     | ORT_PARALLEL   | ORT_ENABLE_BASIC    | 50.02 ms  |
| 8     | ORT_SEQUENTIAL | ORT_ENABLE_BASIC    | 37.93 ms  |
| 8     | ORT_PARALLEL   | ORT_ENABLE_EXTENDED | 46.22 ms  |
| 8     | ORT_SEQUENTIAL | ORT_ENABLE_EXTENDED | 33.76 ms  |
| 8     | ORT_PARALLEL   | ORT_ENABLE_ALL      | 41.16 ms  |
| 8     | ORT_SEQUENTIAL | ORT_ENABLE_ALL      | 30.64 ms  |
| 16    | ORT_PARALLEL   | ORT_DISABLE_ALL     | 102.53 ms |
| 16    | ORT_SEQUENTIAL | ORT_DISABLE_ALL     | 46.80 ms  |
| 16    | ORT_PARALLEL   | ORT_ENABLE_BASIC    | 100.89 ms |
| 16    | ORT_SEQUENTIAL | ORT_ENABLE_BASIC    | 45.88 ms  |
| 16    | ORT_PARALLEL   | ORT_ENABLE_EXTENDED | 99.14 ms  |
| 16    | ORT_SEQUENTIAL | ORT_ENABLE_EXTENDED | 42.59 ms  |
| 16    | ORT_PARALLEL   | ORT_ENABLE_ALL      | 84.10 ms  |
| 16    | ORT_SEQUENTIAL | ORT_ENABLE_ALL      | 38.05 ms  |
*Results on a machine with 8 physical cores and 16 logical cores*

## Result Analysis
First of all, the best results are when we run with 8 intra threads, in sequential order with all optimizations enabled. Our initial thought was that parallel would be faster, because more threading is faster. But there are two likely reasons why this might not be the case in this scenario:
1. The model is quite narrow and sequential as is, meaning gains from running in parallel are quite minimal.
2. At 8 intra threads, and 16 inter threads, we're overscheduling our CPU, causing too much organizational overhead.

Some other notes visible from the data
- The biggest performance gain is when we go from 1 intra thread, to 2 intra threads. This is in line with Amdahl's law.
- When running 1 intra thead, the biggest performance gain we can get is by running the layout graph optimization (aligning data with the CPU), which simplifies and speeds up loading data into SIMD registers, as the data is now stored in the sequence that it's required for the operations.
- We see a shift in performance gain between parallel and sequential as we move up in threads. At 1 to 2 intra threads, running in parallel is a performance gain over sequential. But when we move towards 4+ intra threads, sequential takes the lead.
- At 16 intra threads (where we overstep the amount of physical cores that the machine has), we start seeing an absolutely abysmal performance in parallel processing. At this point, our machine cores will get shared between both intra and inter threads, requiring much more scheduling overhead.

As a final experiment, I tried running the same experiment with an inter thread count of 4 and 2. At 4 inter threads, we mainly see a performance gain in parallel execution with 2 intra threads. This performance gain is 5-6ms on average. At 2 inter threads, we see the most significant gain at 8 intra threads, in parallel with basic optimization. Here we have a performance gain of almost 9ms. This configuration is still a bit more than 10ms short of taking the lead in the best performant configuration.

So in short, we see that the best performance is to stick with physical cores, multi-thread the calculations within a node, run nodes sequentially as to reduce scheduling overhead and trust ONNX to know how to optimize a graph for you. 
