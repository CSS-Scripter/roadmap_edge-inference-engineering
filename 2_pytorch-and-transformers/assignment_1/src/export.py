import os
import torch

from bigram_language_model import BigramLanguageModel
from constants import block_size, device

working_dir  = os.path.dirname(__file__)
dataset_file = os.path.join(working_dir, "../../data/tinyshakespeare.txt")
checkpoint   = os.path.join(working_dir, "../checkpoints/2026-06-09_22-00-1781035206.pt")

with open(dataset_file, 'r', encoding='utf-8') as f:
    text = f.read()

vocab_size = len(list(set(text)))


m = BigramLanguageModel(vocab_size)
m.load_state_dict(torch.load(checkpoint, weights_only=True))
m.eval()

total = sum(p.numel() for p in m.parameters())
print(f"Total parameters: {total:,}")

for name, p in m.named_parameters():
    print(f"{p.numel():>10,}  {name}")

# input_tensor = torch.zeros((1, block_size), dtype=torch.long, device=device)
# torch.onnx.export(
#     m,
#     (input_tensor,),
#     "tiny_gpt.onnx",
#     input_names=["idx"],
#     output_names=["idx_out"],
#     external_data=False,
# )

# print("=" * 100)

# exported = torch.export.export(m, (input_tensor,))
# print(exported)
