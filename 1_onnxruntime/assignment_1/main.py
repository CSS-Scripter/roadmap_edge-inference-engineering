# 
# Assignment 1: Graph Inspector
# Take any model from the table above. Write a script (Python or C++) that:
# - Prints every node: op type, input names, output names, attributes
# - Lists every initializer (weight tensor) with its shape and dtype
# - Identifies which nodes would likely fall back to CPU if you used a non-CPU EP
# 
# Deliverable: a script and a written summary of what you found in your model.
# 

from typing import Any
import onnx
import onnxruntime as ort

from google.protobuf.json_format import MessageToDict

import os


models_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../models"))

modelpath_mobilenet = os.path.join(models_path, "mobilenetv2-7.onnx")
modelpath_yolo      = os.path.join(models_path, "yolov8n.onnx")
modelpath_resnet    = os.path.join(models_path, "resnet50-v1-7.onnx")
    


def main():
    model = onnx.load(modelpath_yolo)
    # model = onnx.load(modelpath_mobilenet)
    # model = onnx.load(modelpath_resnet)

    graph = model.graph

    operators = set()

    for node in graph.node:
        print(f"Node name: {node.name}")
        print(f"OP Type: {node.op_type}")
        print(f"Inputs: {node.input}")
        print(f"Outputs: {node.output}")
        print(f"Attributes: {node.attribute}")

        print("-" * 30)

        operators.add(node.op_type)

    print(f"Unique operators: \n {'\n- '.join(list(operators))}")



if __name__ == "__main__":
    main()


