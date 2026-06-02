import os
import torch
from bigram_language_model import BigramLanguageModel

working_dir = os.path.dirname(__file__)
data_dir    = os.path.join(working_dir, "../../data")

tinyshakespeare_file = os.path.join(data_dir, "tinyshakespeare.txt")

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def main():
    with open(tinyshakespeare_file, "r") as f:
        dataset_text = f.read()
    
    chars = sorted(list(set(dataset_text)))
    vocab_size = len(chars)

    print(f"Vocabulary: {''.join(chars)}")
    print(f"Vocabulary size: {vocab_size}")

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(dataset_text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data[:100])

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]


    batch_size = 4
    block_size = 8

    xb, yb = get_batch(train_data, block_size, batch_size)
    print("inputs:")
    print(xb.shape)
    print(xb)
    print("targets:")
    print(yb.shape)
    print(yb)

    m = BigramLanguageModel(vocab_size)
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)


    idx = torch.zeros((1, 1), dtype=torch.long)
    generated = m.generate(idx, 100)
    print("= WITHOUT TRAINING ==========")
    print(decode(generated[0].tolist()))
    print("===========================")


    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    batch_size = 32
    for steps in range(1000):
        xb, yb = get_batch(train_data, block_size, batch_size)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())

    idx = torch.zeros((1, 1), dtype=torch.long)
    generated = m.generate(idx, 100)
    print("= WITH TRAINING =============")
    print(decode(generated[0].tolist()))
    print("=============================")


    


if __name__ == "__main__":
    main()
