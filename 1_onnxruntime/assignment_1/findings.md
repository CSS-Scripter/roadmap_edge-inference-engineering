```python
import onnx

model = onnx.load(modelpath_yolo)
graph = model.graph

for node in graph.node:
    print(f"Node name: {node.name}")
    print(f"OP Type: {node.op_type}")
    print(f"Inputs: {node.input}")
    print(f"Outputs: {node.output}")
    print(f"Attributes: {node.attribute}")

    print("-" * 30)
```

Node names are shaped as directory paths, with the first segment being `model.X` and the last segment being the operation type (e.g. Mul, Sigmoid, Conv). There are some folders in between which do seem a bit mysterious. It's also unclear what determines the number in the first folder (replacing the X).
Example names:
- /model.0/act/Sigmoid
- /model.1/conv/Conv
- /model.2/cv1/act/Mul
- /model.4/m.0/cv1/conv/Conv

When looking at the names in MobileNet, this structure completely breaks. It seems that naming is mainly for organizing, with the constraint that they need to be unique, since they're also used as identifiers for inputs and outputs of other nodes.


The operator types seem to be one of several possible operators. Adding a quick `set` collection and print, it shows all unique operators in the model are:
- Slice
- Concat
- Softmax
- Div
- Resize
- Mul
- MaxPool
- Add
- Reshape
- Transpose
- Sub
- Sigmoid
- Conv

Some operators are basic mathematical operations, like:  `Div`, `Mul`, `Add` and `Sub` (for division, multiplication, addition and subtraction). An addition to this, that is not in the YOLO model, is `MatMul` and `Gemm`.
- `MatMul` being pure matrix multiplications $C=A*B$.
- `Gemm` stands for General Matrix Multiply, borrowed from `BLAS` (Basic Linear Algebra Subprograms). It calculates as $C=\alpha*A*B+\beta*C$. It has several features that help in fully-connected/dense layers.

Additionally, there is the `Sigmoid` function, which forces a value to be between 0 and 1.


There are also array/matrix manipulation operators, such as:
- `Slice`: Create a subarray from a larger array
- `Concat`: Join two or more arrays together
- `Reshape`: Change the shape of multidimensional arrays
- `Resize`: Resizes arrays to fit a certain format, e.g. changing the resolution of an image. This means data will either be generated, or dropped. The exact way this is done, is through an interpolation mode. Some common ones are `Nearest neighbor`, `Linear/BiLinear` and `Cubic/BiCubic`.
- `Transpose`: Similar to reshape, but is also able to change the order of the data. E.g. changing RGB into BGR. The reshaping may only happen through permuting existing axes, meaning the numbers in the shape must remain, whereas in a reshape, the product of the shapes must remain the same. E.g. (3, 64, 64) -> (64, 64, 3) is allowed, but (3, 64, 64) -> (12, 16, 64) is not, since we originally did not have a 12 or 16 sized axis.
- `SoftMax`: normalize the values of an array, so that the sum of the array is 1.

Finally, there are neural operators:
- `Conv`: Convolutional layer. It uses a hyperparameter for size (e.g. 3x3), and has several kernels in that shape. Each kernel simply contains a matrix of weights of the size in the hyperparameter, and is used to slide across the input matrix, to get an activation matrix out.

- `MaxPool`: Takes the max value out of a given window. It looks within a window (called a kernel) of a given size, and moves by a value called stride. Going over a matrix of (10,10) with a size of 2x2 and a stride of 1 will result in (9,9). Though a size of 2x2 and stride of 2 will result in (5,5). 


Some operatores used in MobileNet, that are not mentioned above are:
- `BatchNormalization`: Normalizes the values within a batch to have a mean of 0, and a standard deviation of 1.
- `ReLu`: Rectified Linear Unit, a linear activation unit. Sets the minimum value of an activation to 0. Comparable to `math.max(activation, 0)` in most programming languages.
- `GlobalAveragePool`: Takes the average across the full feature window. (64,64,3) becomes (64,1,1). 

Operators in ResNet, not in Yolo or MobileNet:
- `Flatten`: flattens all dimensions into a single dimension. So (64,64,3) becomes (12288,)
- `Gemm`: General Matrix Multiply, borrowed from `BLAS` (Basic Linear Algebra Subprograms). It calculates as $C=\alpha*A*B+\beta*C$. It has several features that help in fully-connected/dense layers.


Finally there are the attributes, which we can think of as parameters to the different steps within the operator. For example, the attributes to a Conv operator:
```
[
    name: "dilations"
        ints: 1
        ints: 1
        type: INTS,

    name: "group"
        i: 1
        type: INT,

    name: "kernel_shape"
        ints: 1
        ints: 1
        type: INTS,

    name: "pads"
        ints: 0
        ints: 0
        ints: 0
        ints: 0
        type: INTS,

    name: "strides"
        ints: 1
        ints: 1
        type: INTS
]
```

For example, we mentioned in the explanation of the Conv operator that the kernal shape was passed through a "hyperparameter". This is it. We have a parameter called "kernel_shape" with values `ints: 1 ints: 1`, meaning 1x1 kernels. The four ints in `pads` are for padding the different sides of the input. Strides being the steps we take, also 1 in each direction.

These attributes, also called `hyperparameters` are baked into the model, and are decided at model design time.


---

Initializers can be printed in a similar fashion as the nodes.
```python
def main():
    model = onnx.load(modelpath_yolo)

    graph = model.graph

    for initializer in graph.initializer:
        dtype = onnx.TensorProto.DESCRIPTOR.EnumValueName("DataType", initializer.data_type)

        print(f"Initializer name: {initializer.name}")
        print(f"Dimensions: {initializer.dims}")
        print(f"DType: {dtype}")

        print("-" * 30)
```

In the case of the YOLO model, we can quite easily relate initializers to the operators that use them. For example, given the following operator:
```
Node name: /model.0/conv/Conv
OP Type: Conv
Inputs: ['images', 'model.0.conv.weight', 'model.0.conv.bias']
Outputs: ['/model.0/conv/Conv_output_0']
```

It uses two initializers: `model.0.conv.weight` and `model.0.conv.bias`.

We can then check these initializers:
```
Initializer name: model.0.conv.weight
Dimensions: [16, 3, 3, 3]
DType: FLOAT
```

```
Initializer name: model.0.conv.bias
Dimensions: [16]
DType: FLOAT
```

And considering the hyperparameter `kernel_shape` of this Conv operator is 3x3, we can deduce that we're looking at 16 different kernels, across 3 channels, with a 3x3 shape, resulting in the weight shape of `[16,3,3,3]`.
We then apply a bias to each result. The bias simply contains 16 values, 1 for each kernel. So we offset the result of each kernel by the bias, and that is the Conv operation done.

---

CPU Fallback is a bit trickier, because you need to find out what hardware should run the code, and what operators they support. Generally this should be pretty well documented online. In my case, I looked at the CUDA Execution Provider. They have their supported operators listed on [Github](https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md).

Using a function to scrape this markdown file (thanks Claude), we can start comparing it against the operators in the models, as well as the types used for the input into those operators.

```python
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
```

Next, we need to find the types used in our operators. Looking at how we currently print inputs to each operator, there seems to be no way to find what the actual dtype of this value is. Instead, we need to look at the graph's value_info.

```python
def main():
    model = onnx.load(modelpath_mobilenet)
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
```

With this collection of input, output and initializers, along with their types and shapes, we should be able to quite confidently say which types are used for which operators. We can then match this against the CUDA kernel documentation, to find our fallbacks.

```python
    # AGGREGATING OPERATOR INFORMATION

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


    # SUMMARIZING OPERATOR INFORMATION

    unique_operators = list(set([op["name"] for op in operators]))
    operator_inputs = {op: set([]) for op in unique_operators}
    for operator in operators:
        operator_inputs[operator["name"]] = set(list(operator_inputs.get(operator["name"], [])) + operator["input_types"])

    print("Operator overview")
    for op, inputs in operator_inputs.items():
        print(f"{op}: {list(inputs)}")


    # FINDING FALLBACK CANDIDATES

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
```
