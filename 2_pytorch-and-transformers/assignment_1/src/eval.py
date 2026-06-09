import os
import torch
from bigram_language_model import BigramLanguageModel
from constants import device


working_dir  = os.path.dirname(__file__)
data_dir     = os.path.join(working_dir, "../../data")
dataset_file = os.path.join(data_dir, "tinyshakespeare.txt")


with open(dataset_file, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



m = BigramLanguageModel(vocab_size)
m.load_state_dict(torch.load("checkpoints/shakespear/2026-06-06_00-31-1780698675.pt", weights_only=True))
m.eval()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
for _ in range(1000):
    context = m.generate(context, max_new_tokens=1)
    print(decode(context[0][-1:].tolist()), end="")

# print(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))
