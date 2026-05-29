# Assignment 1: Static Analysis

> 
> Take any model from the table above. Write a script (Python or C++) that:
> 1. Prints every node: op type, input names, output names, attributes
> 2. Lists every initializer (weight tensor) with its shape and dtype
> 3. Identifies which nodes would likely fall back to CPU if you used a non-CPU EP
> Deliverable: a script and a written summary of what you found in your model.
> 

## Node analysis

Nodes are the computational building blocks of an ONNX model. They take in data, run calculations on it, and output data. Although they haven't shown up in the analyzed models, it's likely also the case that there are operators that do not take in data, but do give an output, for example to generate a sin-wave. Outputs from one node, are the inputs to the next node. 

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
*python script that prints all nodes in a model, including their name, operator type, inputs, outputs and attributes*

`node.name` is a developer defined field, that's used across the graph to indicate inputs and outputs. `node.input` is a list of strings that indicate the name's of the sources that are used as input for this node. `node.output` is a list of strings that show the names of the output values, which can then get referenced for node inputs.

`node.op_type` shows what operation the node will run on the inputs to create the outputs. `node.attribute` shows all the hyper parameters for these operators. A hyper parameter is a value that's decided at design time, and functions as a configuration for the operator. More examples will be given under operator types.

### Operator Types

Looking through the models YOLOv8n, MobileNet and ResNet50, we have found the following ONNX operator types: Slice, Concat, Softmax, Div, Resize, Mul, MaxPool, Add, Reshape, Transpose, Sub, Sigmoid, Conv, BatchNormalization, ReLu, GlobalAveragePool, Flatten and Gemm. An operator not in this list, but worth mentioning is MatMul.

Some operators will be quite straightforward, like `Add`, `Sub`, `Div` and `Mul`. These are mathematical operations to be done against 2 tensors. A constraint on this is that the shapes must match from the right side, or be 1. So valid options for A and B as inputs are `A: (3, 4, 5) B: (4, 5)` and `A: (3, 4, 5) B: (1, 5)`, but this is invalid: `A: (3, 4, 5) B: (3, 5)`.

Another mathematical operator is the `Sigmoid` operator, which applies the sigmoid function to values, mapping them to a value between 0 and 1.

Then there is `MatMul` and `Gemm`, for matrix multiplication. `MatMul` is the pure matrix multiplication, notated as $C=A*B$. `Gemm` stands for General Matrix Multiply, borrowing from `BLAS` (Basic Linear Algebra Subprograms). It's notation is $C = \alpha * A * B + \beta * C$. There are several features to `Gemm` that help in dense and fully connected models.

Then there are array operators, which you might recognize from other programming languages. There are `Slice`, `Concat`, `Reshape`, `Flatten`, `Resize` and `Transpose`.
`Slice` and `Concat` of course for splitting an array in two, and joining two arrays into one. `Reshape` and `Flatten` for changing the dimensions on a tensor. `Resize` also fits into this changing of dimensions idea, with the difference that `Resize` may generate or drop samples within a tensor. The generation may be done through one of many interpolation modes, with the most common being `Nearest Neighbor` (copy the closest value), `Linear/BiLinear` (linear interpolation) and `Cubic/BiCubic` (cubic interpolation). 

`Transpose` is able to shift around the dimensions of a tensor, altering the order in which the data is shown. This is particularly useful when you need to abide by a certain format. A common one, and the one YOLOv8n uses, is called `NCHW`, or `batch, channel, height, width` and in this models case has a shape of `(1, 3, 640, 640)`. We might have to change this into `NHWC`, which would then be shaped `(1, 640, 640, 3)`. But to actually abide by this standard, we can't just resize, because that would take the first 3 pixels, and put them as separate channels. Let's say we have an image, that is entirely green (`#0F0`). All values in the green channel are 1's, while the values in the red and blue channels are 0's. If we simply resize, we get an image that is black, with the middle third of the image being green. Whereas if we use transpose, the image stays green, because the order in data is adjusted to fit the new format. This adjustment in format imposes a limitation of `tranpose`, which is that the dimensions must stay the same size, and can only get reordered. Meaning that original shape of `(1, 3, 640, 640)` can't get changed into `(1, 640, 320, 6)`, since it wouldn't be able to determine how to reorder the data.

Finally that are the analytical and machine learning operators, such as `SoftMax`, `Conv`, `MaxPool`, `BatchNormalization`, `ReLu` and `GlobalAveragePool`. The two simplest being `ReLu`, which is comparable to setting a minimum value of 0 on values, or for programmers, it functions as `math.max(v, 0)`. The second simplest is the `SoftMax`, which normalizes the values of an array to sum to 1.

`BatchNormalization` is another statistical operator. It shifts and stretches the values in an array so that the mean value becomes 0, and the standard deviation becomes 1. During training it learns the Gamma (stretch factor) and Beta (shift amount). During training, it also keeps track of the running mean, and running variance, which are then frozen at export, and used for an optimized calculation during inference. During inference, these values are all pinned and used to get a mean of 0, and a standard deviation of around 1, though through training the network might decide a different value works better. Example: input `[-150, -100, -60, -50, -40, 10, 40]`. Apply a beta of 50, turns into `[-100, -50, -10, 0, 10, 50, 100]`. Now that the mean is 0, we can stretch the values so that the standard deviation is 1, and we end up with `[-1.75, -0.88, -0.18, 0, 0.18, 0.88, 1.75]`.

`Conv`, also called the `Convolution` operator, applies a sliding window of weights to a feature map called a kernel. How you can imagine this, is to compare it with how image blurring works. When you blur an image, you look at each pixel, and take the average color of that and it's surrounding pixel, blending them together. When done to all pixels in an image, we end up with a blurred version of this image. The `Conv` operator would work in a similar fashion, but instead having each pixel count equally, we have trained weights that decide how we should look at each pixel within this sliding window. The actual size of this window, and the amount that it moves (also called `stride`) are decided through `hyper parameters`. If we look at the first `Conv` operator in our model, we see the following hyper parameters
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
		ints: 3
		ints: 3
		type: INTS,

	name: "pads"
		ints: 1
		ints: 1
		ints: 1
		ints: 1
		type: INTS,

	name: "strides"
		ints: 2
		ints: 2
		type: INTS
]
```
Here we see that we're working with a kernel shape of 3x3, and a stride of (2, 2). Meaning that we'll get an output, roughly half the size of the input feature map. In this case, the input is shaped `[1, 3, 640, 640]` (NCHW format), and the kernel weights are shaped as `[16, 3, 3, 3]`, meaning we're applying 16 kernels to the input. The output of this will be shaped as `[1, 16, 320, 320]`, which might leave you wondering: where did that channels dimension go to? We actually took the dot product across the channels, and combined all three channels into a single value.

To kind of summarize the results from a `Conv` operator, we can apply Pool operators. We found the `MaxPool` and `GlobalAveragePool` operators. `MaxPool` slides a window of a given size (e.g. 2x2) across a feature map, and records the highest activation within that region. The `GlobalAveragePool` discards this sliding window idea, and simply takes the average across the entire feature map. So if we were to apply the `GlobalAveragePool` operator to the output of the above mentioned `Conv` operator (output shaped `[1, 16, 320, 320]`), then we'd get an output shaped as `[1, 16, 1, 1]`.

### Analyzing node outputs
ONNX doesn't actually have the output shapes of each operator stored within the model. So to take a look at this, we need to do a little preparation of the model, and look somewhere else for this information.

```python
import onnx

model = onnx.load(modelpath_yolo)
model = onnx.shape_inference.infer_shapes(model)
graph = model.graph

for value_info in graph.value_info:
	dtype = onnx.TensorProto.DESCRIPTOR.EnumValueName(
		"DataType",
		value_info.type.tensor_type.elem_type
	)
	shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
	
	print(f"Name: {value_info.name}")
	print(f"DType: {dtype}")
	print(f"Shape: {shape}")

	print("-" * 30)
	
```

The name of this `value_info` object will correspond to the names mentioned in the `input` and `output` fields of our graph nodes.

## Initializer Analysis

Initializers are, similar to node outputs, values that we can use for node inputs, with the difference being that initializers are not node outputs. Initializers are often trained values, like convolution weights and biases. Though it may also occur that the output of node a during evaluation is static, causing it to be exported as an initializer as to optimize the model. 

```python
import onnx

model = onnx.load(modelpath_yolo)
graph = model.graph

for initializer in graph.initializer:
	dtype = onnx.TensorProto.DESCRIPTOR.EnumValueName(
		"DataType",
		initializer.data_type
	)

	print(f"Initializer name: {initializer.name}")
	print(f"Dimensions: {initializer.dims}")
	print(f"DType: {dtype}")

	print("-" * 30)
```

The above script will print out all initializers in the YOLO model. Due to the structured naming of nodes and initializers within the YOLO model, we can actually quite easily identify which initializer is used where, and which ones are actually one of those "optimized" node output initializers, since those will have "output" in their name.

Let's take a look at some initializers used in that first `Conv` node:

```
--------------------------------------------
Node name: /model.0/conv/Conv
OP Type: Conv
Inputs: ['images', 'model.0.conv.weight', 'model.0.conv.bias']
Outputs: ['/model.0/conv/Conv_output_0']
--------------------------------------------
Initializer name: model.0.conv.weight
Dimensions: [16, 3, 3, 3]
DType: FLOAT
--------------------------------------------
Initializer name: model.0.conv.bias
Dimensions: [16]
DType: FLOAT
--------------------------------------------
```

Now `images` is not an initializer, but a model input. We need a slightly different script to view that one, but it works similar to what we've seen so far.
```python
for i in graph.input:
	dtype = onnx.TensorProto.DESCRIPTOR.EnumValueName(
		"DataType",
		i.type.tensor_type.elem_type
	)
	shape = [dim.dim_value for dim in i.type.tensor_type.shape.dim]

	print(f"Name: {i.name}")
	print(f"DType: {dtype}")
	print(f"Shape: {shape}")
```
*to print the model outputs, simply look at `graph.output`*

This will show that `images` is shaped as `[1,3,640,640]` with a float dtype. We apply a `Conv` operator to this, using the weights shaped `[16,3,3,3]`, got the dot product from the results to merge channel dimensions, and applied a bias to each feature map, simply by adding the right value from the bias initializer to the right feature map.

Some initializers also have a shape of `[]`. This simply indicates that the value is not an array, but a single value. This is not shown as `[1]` because that would imply a single value inside an array, while the value of the initializer is not in an array.

## CPU Fallbacks

CPU Fallbacks happen when we try to run an operator on specific hardware (that's not a CPU), but the execution provider for that hardware is missing kernels to run this operator. In that scenario, that one specific operator will fallback onto the CPU, and run there.

A bit about finding fallback operators is that it can be done through 2 methods: static analysis and profiling. With static analysis, we don't actually run the model, but rather estimate, so it may contain mistakes. As for profiling, we will literally run the model with an execution provider, and see if it falls back or not, so this approach is much closer to the truth than static analysis. In this scenario we'll use static analysis, which may produce false negatives. In a later assignment (assignment 3) we will look at the profiler. 

So before we can check this, we first need to decide: on which hardware are we running, and then try and find which operators are supported for this hardware. In my case, I decided to go with the `CUDAExecutionProvider` for NVIDIA GPU's. With a bit of research, I found the [list of supported operators](https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md) on Github.

Let's start by scraping this markdown file for the supported operators and their input types. Or in other words, kindly ask Claude to do this because I hate writing crawlers and parsers, and in other scenario's, I might as well go the profiling analysis route.
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

Next, I rewrote the script so far a bit, so it neatly shows the types and shapes of each possible input and output. This way, we can find all types each operator is called with, and at the final step check if the operator is supported, and if all types used inside the operator are supported.

```python
import onnx

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

unique_operators = list(set([op["name"] for op in operators]))
operator_inputs = {op: set([]) for op in unique_operators}
for operator in operators:
	operator_inputs[operator["name"]] = set(list(operator_inputs.get(operator["name"], [])) + operator["input_types"])

print("Operator overview")
for op, inputs in operator_inputs.items():
	print(f"{op}: {list(inputs)}")
	
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

When we run this, we will find that all operators should be able to run on GPU, as both the operator types, as well as their input types are supported by the CUDA execution provider.
