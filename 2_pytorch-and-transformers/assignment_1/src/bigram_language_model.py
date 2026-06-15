import torch
import torch.nn as nn
from torch.nn import functional as F

from constants import n_embed, block_size, device, n_layer, dropout, n_head


class MQHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, k, v):
        B,T,C = x.shape
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # type: ignore
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class MultiQueryAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.value = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.heads = nn.ModuleList([MQHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed, device=device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        k = self.key(x)
        v = self.value(x)

        out = torch.cat([h(x, k, v) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class GroupedQueryAttention(nn.Module):
    def __init__(self, num_heads, group_size, head_size):
        super().__init__()
        self.num_groups = num_heads // group_size
        self.keys = nn.ModuleList([nn.Linear(n_embed, head_size, bias=False, device=device) for _ in range(self.num_groups)])
        self.values = nn.ModuleList([nn.Linear(n_embed, head_size, bias=False, device=device) for _ in range(self.num_groups)])
        self.heads = nn.ModuleList([MQHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ks = [self.keys[i](x) for i in range(self.num_groups)]
        vs = [self.values[i](x) for i in range(self.num_groups)]

        out = torch.cat([
            h(
                x,
                ks[i%self.num_groups],
                vs[i%self.num_groups]
            )
            for i, h in enumerate(self.heads)
        ], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.query = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.value = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # type: ignore
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed, device=device),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed, device=device),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed, device=device)
        self.ln2 = nn.LayerNorm(n_embed, device=device)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed, device=device)
        self.position_embedding_table = nn.Embedding(block_size, n_embed, device=device)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embed, device=device)
        )
        self.lm_head = nn.Linear(n_embed, vocab_size, device=device)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# B = Batch
# T = Time
# C = Channel
