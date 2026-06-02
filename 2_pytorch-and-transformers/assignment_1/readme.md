# Assignment 1: Creating Karpathy's GPT

> 
> Follow Karpathy's GPT from scratch video and implement a character-level language model. Type the code yourself — don't copy-paste.
> 
> Extend it:
> 
> Add a function that prints the shape of every tensor at each stage of a single forward pass
> Add a comment on every major operation explaining what it's doing to the data shape
> 

Resources: [TinyShakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)


# Encoder

Given our dataset, we can find the unique characters in this dataset, along with the total vocabulary size.

```python
chars = sorted(list(set(dataset_text)))
vocab_size = len(chars)

print(f"Vocabulary: {''.join(chars)}")
print(f"Vocabulary size: {vocab_size}")
```

We can assign an integer to each of these characters, and use it to "encode" and "decode" a string.

```python
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

Of course, this is a very simple encoding and decoding of a string to a list of integers and back. Nothing more than what Python already does under the hood with ASCII and UTF-8. But there are other ways to do this, for example using Google's [SentencePiece](https://github.com/google/sentencepiece) library, which uses sub-word units. There's also OpenAI's [TikToken](https://github.com/openai/tiktoken). But for now, this simple approach is easy to understand, and will work fine.

Let's encode our data into a tensor, and split it into a training, and valuation dataset.

```python
import torch

data = torch.tensor(encode(dataset_text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
```

Now the goal is to a structure in the data, that gives a sequence of these encoded characters, and is able to say what the next character is. A bit like this:

```python
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the targets: {target}")
```

We create 8 arrays, increasing by 1 character each time, and say: for this sequence of characters, we want to see the next character. To do this at a bit larger scale across the whole dataset, we can use the following function

```python
def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
```

What this function does, is it creates `batch_size` amount of the above mentioned examples. It does this by first creating random starting indices `torch.randint(len(data)-block_size, (batch_size,))`. It then creates tensors with these datasets, where x is the context, and y is the target. For every `i`, the case is `for context x[:i] we expect y[i]`. The stack simply adds another array, so we get `for context x[batch][:i] we expect y[batch][i]`.

```
when input is tensor([18]) the targets: 47
when input is tensor([18, 47]) the targets: 56
when input is tensor([18, 47, 56]) the targets: 57
when input is tensor([18, 47, 56, 57]) the targets: 58
when input is tensor([18, 47, 56, 57, 58]) the targets: 1
when input is tensor([18, 47, 56, 57, 58,  1]) the targets: 15
when input is tensor([18, 47, 56, 57, 58,  1, 15]) the targets: 47
when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the targets: 58
```

Now to actually make a model that will consume this data

```python
import torch.nn as nn

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        return logits
```

In the initializer, we create an embedding table, of vocab_size x vocab_size (or 65x65). This table essentially is a single-character lookup table of what the most likely next character is, based on a single character. Since every row, and every column maps to a unique character, we can use this character to get a slice of probabilities, which contains one probability per character.

Of course, targets are used to find the loss, which we'll use later on to determine how we should train the transformer. So let's add loss, but also make targets optional so we can use it for generation as well.

```python
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

Here we see the deconstruction of the shape of logits as well. The characters B, T and C represent Batch, Time and Channel.

Now we can generate text, by looping over this forward function, and concatenating the most probable character to the end of our idx, and looping until we're complete.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

Feed it some random data, and bang, large language model.

```python
m = BigramLanguageModel(vocab_size)
idx = torch.zeros((1, 1), dtype=torch.long)
generated = m.generate(idx, 100)
print(decode(generated[0].tolist()))
```

Of course the output of this piece of code is not great. It's absolutely nonsense random characters. And that's because we haven't trained the model yet. Let's bring in the optimizer, and train the model.

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(10000):
    xb, yb = get_batch(train_data, block_size, batch_size)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

Now, when we run the generation again, it will be much more representative of text. Still meaningless, but more like english.

```
== BEFORE ====
m,&BFoa3Niq?ak$k$bATL3R$k$JUPnT'ruhLODM:&ZuIOpfFDFr
3-JAPzVJTIudkbAVJUdQqogiI!g3-mvSNnjSjCcTvKJa$qKk

== AFTER  ====
My, chirimeesuru; alfopavend INThet sorene ck my l Wadsousecoured maf,

The, haindise; blf. helexe s
```

This is of course trained and ran while only looking at the last token of the context. But how do we get context about what was before a token? A very simple, but weak and lossy, method is to just take the average of all the preceding tokens.

We can get the average of all tokens in a matrix very efficiently through matrix multiplications.

```python
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print('a=')
print(a)
print('---')
print('b=')
print(b)
print('---')
print('c=')
print(c)
print('---')

# a=
# tensor([[1.0000, 0.0000, 0.0000],
#         [0.5000, 0.5000, 0.0000],
#         [0.3333, 0.3333, 0.3333]])
# ---
# b=
# tensor([[9., 2.],
#         [5., 5.],
#         [1., 2.]])
# ---
# c=
# tensor([[9.0000, 2.0000],
#         [7.0000, 3.5000],
#         [5.0000, 3.0000]])
```

Now we see that `c[2,0]` is the average of `b[:2,0]`, because we created a smart multiplication matrix in A, that has the amount a single value should count for the average per row.


(left of at 1:18:40)
