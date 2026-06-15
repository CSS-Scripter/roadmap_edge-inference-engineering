import time
from transformers import pipeline

gpt2_pipe = pipeline("text-generation", model="gpt2")
t5_pipe = pipeline("text2text-generation", model="t5-base")
bert_pipe = pipeline("feature-extraction", model="bert-base-uncased")

inputs = {
    "gpt2": "Write a poem. Poet:",
    "t5": "Write a poem:",
    "bert": "Write a poem about the nature of time and its effect on human consciousness."
}

runs = 5  # average over multiple runs to reduce noise

for name, pipe, prompt in [
    ("GPT-2", gpt2_pipe, inputs["gpt2"]),
    ("T5",    t5_pipe,   inputs["t5"]),
    ("BERT",  bert_pipe, inputs["bert"]),
]:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        if name == "GPT-2":
            pipe(prompt, max_new_tokens=200)
        elif name == "T5":
            pipe(prompt, max_new_tokens=200)
        else:
            pipe(prompt)
        times.append(time.perf_counter() - start)
    avg = sum(times) / runs
    print(f"{name}: {avg:.3f}s avg over {runs} runs")
