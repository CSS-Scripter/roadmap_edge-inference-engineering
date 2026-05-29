#
# Write a benchmarking harness that runs inference 100 times and measures average latency. Vary:
# 
# - Intra-op thread count (1, 2, 4, 8, max)
# - Graph optimization level (all four levels)
# - Execution mode (sequential vs parallel)
# 
# Table the results. Write a short explanation of which settings performed best and why.
#

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
    cores = psutil.cpu_count(logical=False)
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
    session_options.inter_op_num_threads = 2
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
