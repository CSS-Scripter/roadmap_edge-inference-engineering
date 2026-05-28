# 
# Assignment 1: Graph Inspector
# Take any model from the table above. Write a script (Python or C++) that:
# - Prints every node: op type, input names, output names, attributes
# - Lists every initializer (weight tensor) with its shape and dtype
# - Identifies which nodes would likely fall back to CPU if you used a non-CPU EP
# 
# Deliverable: a script and a written summary of what you found in your model.
# 

import requests
import re

def fetch_cuda_supported_ops():
    url = "https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/OperatorKernels.md"
    response = requests.get(url)
    content = response.text

    # Find the CUDA section identifier from the TOC
    toc_match = re.search(r'\[CUDAExecutionProvider\]\(#([^)]+)\)', content)
    if not toc_match:
        return {}
    identifier = toc_match.group(1)

    # Find the section using the identifier
    section_match = re.search(
        rf'<a name="{re.escape(identifier)}"[^>]*/>\s*\n+##[^\n]+\n+(.*?)(?=\n<a name="|$)',
        content,
        re.DOTALL
    )
    if not section_match:
        return {}

    section = section_match.group(1)

    # Extract operator names from the first column of the markdown table
    ops = {}
    for line in section.splitlines():
        line = line.replace("<br>", "").replace("<br/>", "")
        match = re.match(r'\|*(\w+)\|.*\|.*\|(.*)\|', line)
        if match:
            operator = match.group(1)
            supported_types = ', '.join(re.findall(r'tensor\(([\w\d]*)\)', match.group(2)))
            supported_types = supported_types.replace("tensor(", "")
            supported_types = supported_types.replace(")", "")
            supported_types = [st.upper() for st in supported_types.split(", ")]
            ops[operator] = supported_types

    return ops


# =====================================

from typing import Any
import onnx
import onnxruntime as ort

import os


models_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../models"))

modelpath_mobilenet = os.path.join(models_path, "mobilenetv2-7.onnx")
modelpath_yolo      = os.path.join(models_path, "yolov8n.onnx")
modelpath_resnet    = os.path.join(models_path, "resnet50-v1-7.onnx")
    


def main():
    # model = onnx.load(modelpath_yolo)
    model = onnx.load(modelpath_mobilenet)
    # model = onnx.load(modelpath_resnet)

    model = onnx.shape_inference.infer_shapes(model)

    graph = model.graph
    shape_inference = {}
    for vi in graph.value_info:
        dtype = onnx.TensorProto.DESCRIPTOR.EnumValueName("DataType", vi.type.tensor_type.elem_type)
        shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
        shape_inference[vi.name] = { "dtype": dtype, "shape": shape }

    for initializer in graph.initializer:
        dtype = onnx.TensorProto.DESCRIPTOR.EnumValueName("DataType", initializer.data_type)
        shape = [dim for dim in initializer.dims]
        shape_inference[initializer.name] = { "dtype": dtype, "shape": shape }

    for i in graph.input:
        dtype = onnx.TensorProto.DESCRIPTOR.EnumValueName("DataType", i.type.tensor_type.elem_type)
        shape = [dim.dim_value for dim in i.type.tensor_type.shape.dim]
        shape_inference[i.name] = { "dtype": dtype, "shape": shape }

    for o in graph.output:
        dtype = onnx.TensorProto.DESCRIPTOR.EnumValueName("DataType", o.type.tensor_type.elem_type)
        shape = [dim.dim_value for dim in o.type.tensor_type.shape.dim]
        shape_inference[o.name] = { "dtype": dtype, "shape": shape }


    operators = []
    for node in graph.node:
        inputs = []
        for input in node.input:
            if input == "":
                continue
            inputs.append({
                "name": input,
                **shape_inference.get(input, {})
            })

        outputs = []
        for output in node.output:
            outputs.append({
                "name": output,
                **shape_inference.get(output, {})
            })

        input_types = [inp.get("dtype", None) for inp in inputs]
        operators.append({
            "name": node.op_type,
            "input_types": input_types,
        })

        print(f"Node name: {node.name}")
        print(f"OP Type: {node.op_type}")
        print(f"Inputs: {inputs}")
        print(f"Outputs: {outputs}")
        print(f"Attributes: {node.attribute}")

        print("-" * 30)

    print("=" * 30)

    unique_operators = list(set([op["name"] for op in operators]))
    operator_inputs = {op: set([]) for op in unique_operators}
    for operator in operators:
        operator_inputs[operator["name"]] = set(list(operator_inputs.get(operator["name"], [])) + operator["input_types"])

    print("Operator overview")
    for op, inputs in operator_inputs.items():
        print(f"{op}: {list(inputs)}")


    print("=" * 30)

    print("CPU Fallback Candidates:")
    cuda_supported = fetch_cuda_supported_ops()
    for op, dtypes in operator_inputs.items():
        sts = cuda_supported.get(op, [])

        reasons = []
        if op not in cuda_supported:
            reasons.append("op not supported by CUDA EP")
        for dtype in dtypes:
            if dtype not in sts:
                reasons.append(f"{dtype} input not supported")
        if reasons:
            print(f"{op}: {', '.join(reasons)}")



if __name__ == "__main__":
    main()


