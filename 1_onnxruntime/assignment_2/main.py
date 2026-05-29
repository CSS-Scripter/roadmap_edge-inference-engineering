#
# Write a benchmarking harness that runs inference 100 times and measures average latency. Vary:
# 
# - Intra-op thread count (1, 2, 4, 8, max)
# - Graph optimization level (all four levels)
# - Execution mode (sequential vs parallel)
# 
# Table the results. Write a short explanation of which settings performed best and why.
#

import os
import onnxruntime as ort
import numpy as np
import psutil


models_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../models"))

modelpath_mobilenet = os.path.join(models_path, "mobilenetv2-7.onnx")
modelpath_yolo      = os.path.join(models_path, "yolov8n.onnx")
modelpath_resnet    = os.path.join(models_path, "resnet50-v1-7.onnx")


def get_inter_thread_options():
    """get options for num_inter_threads, without burning out the CPU"""
    physical_cores = psutil.cpu_count(logical=False)
    if physical_cores is None:
        physical_cores = 8 # at this point just guess

    inter_thread_opts = []
    threads = 1
    while threads <= physical_cores:
        inter_thread_opts.append(threads)
        threads *= 2

    return inter_thread_opts


def main():
    inter_thread_opts = get_inter_thread_options()

    session = ort.InferenceSession(
        modelpath_yolo,
    )

    return



if __name__ == "__main__":
    main()
