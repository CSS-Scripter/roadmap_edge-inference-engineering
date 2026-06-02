# ONNX Profiling assignment

> Enable the ONNX Runtime profiler on a model run. From the JSON output, identify:
> 
> - The top 3 operators by total execution time
> - Whether those operators ran on CPU or another EP
> - Any operators that surprised you
> 
> Write a short paragraph explaining what you'd investigate next to make this model faster.


```python
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
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

image = np.random.randn(*session.get_inputs()[0].shape).astype(np.float32)

session.run(None, {"images": image})
session.run(None, {"images": image})
```

We enable profiling by setting `enable_profiling` to true in the session options. This will create a json file, contianing information about each node and session run. Nodes contain data like the duration and timestamp, as well as usefull information under args, like input and output shapes and types, but also the Execution provider. This is the most certain way to tell if a model falls back on another provider or not (other than outright throwing errors and crashing).
