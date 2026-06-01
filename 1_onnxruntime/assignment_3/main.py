# Enable the ONNX Runtime profiler on a model run. From the JSON output, identify:
# 
# - The top 3 operators by total execution time
# - Whether those operators ran on CPU or another EP
# - Any operators that surprised you
# 
# Write a short paragraph explaining what you'd investigate next to make this model faster.


import os
import time

import onnxruntime as ort
import numpy as np

from tabulate import tabulate


models_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../models"))

modelpath_mobilenet = os.path.join(models_path, "mobilenetv2-7.onnx")
modelpath_yolo      = os.path.join(models_path, "yolov8n.onnx")
modelpath_resnet    = os.path.join(models_path, "resnet50-v1-7.onnx")


def main():
    model = modelpath_yolo

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 8
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session_options.enable_profiling = True

    session = ort.InferenceSession(
        model,
        sess_options=session_options,
        providers=[
            ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache"
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )

    image = np.random.randn(*session.get_inputs()[0].shape).astype(np.float32)

    session.run(None, {"images": image})

    start = time.perf_counter()
    session.run(None, {"images": image})
    end = time.perf_counter()

    print(f"duration: {(end-start)*1000:.2f} ms")


if __name__ == "__main__":
    main()
