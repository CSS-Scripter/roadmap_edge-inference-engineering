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
