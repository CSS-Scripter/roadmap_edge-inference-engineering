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


def print_model_metadata(m: onnx.ModelProto):
    print(f"doc_string={m.doc_string}")
    print(f"domain={m.domain}")
    print(f"ir_version={m.ir_version}")
    print(f"metadata_props={m.metadata_props}")
    print(f"model_version={m.model_version}")
    print(f"producer_name={m.producer_name}")
    print(f"producer_version={m.producer_version}")


def format_tensor_dict(tensor_dict: dict[str, Any]):
    formatted_dimensions = []
    dimensions = tensor_dict['type']['tensorType']['shape']['dim']
    for dim in dimensions:
        formatted_dimensions.append(dim.get('dimParam', dim.get('dimValue')))

    return f"name={tensor_dict['name']} elemType={tensor_dict['type']['tensorType']['elemType']} shape=[{','.join(formatted_dimensions)}]"


def print_model_inputs(m: onnx.ModelProto):
    print("inputs: ")
    for i in m.graph.input:
        print(f"  - {format_tensor_dict(MessageToDict(i))}")


def print_model_outputs(m: onnx.ModelProto):
    print("outputs: ")
    for o in m.graph.output:
        print(f"  - {format_tensor_dict(MessageToDict(o))}")


def main():
    model = onnx.load(modelpath_yolo)
    # model = onnx.load(modelpath_mobilenet)
    # model = onnx.load(modelpath_resnet)
    print_model_metadata(model)
    print_model_inputs(model)
    print_model_outputs(model)


if __name__ == "__main__":
    main()
