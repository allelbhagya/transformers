from dataclasses import dataclass
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    # training
    batch_size: int = 32
    max_iters: int = 2000
    eval_interval: int = 100
    eval_iters: int = 200
    learning_rate: float = 1e-3

    # model
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.01

    # extra
    data_path: str = "tiny.parquet"
    seed: int = 1337
    device: str = "cpu"
    vocab_size: int = 0

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def load_data(cfg):
    dataset = load_dataset("parquet", data_files=cfg.data_path, split="train")
    text = "\n\n".join(dataset["text"])
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return tokenizer, data[:n], data[n:]


def get_batch(data, cfg):
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_size = cfg.n_embd // cfg.n_head
        self.block_size = cfg.block_size
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            keep = self.block_size - T
            past_k = past_k[:, :, -keep:]
            past_v = past_v[:, :, -keep:]
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_kv = (k, v)

        att = (q @ k.transpose(-2, -1)) * self.head_size ** -0.5

        # when first call, past_kv is None
        if past_kv is None:
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        att = self.attn_drop(F.softmax(att, dim=-1))
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        return out, new_kv


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None, past_kvs=None):
        B, T = idx.shape
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)
        past_len = min(past_len, self.cfg.block_size - T)
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, past_kv)
            new_kvs.append(new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))

        return logits, loss, new_kvs


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg)
            _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    for it in range(cfg.max_iters):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            l = estimate_loss(model, train_data, val_data, cfg)
            print(f"step {it}: train loss {l['train']:.4f}, val loss {l['val']:.4f}")
        
        x, y = get_batch(train_data, cfg)
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def generate(model, idx, max_new_tokens, cfg):
    model.eval()
    idx_cond = idx[:, -cfg.block_size:]
    logits, _, past_kvs = model(idx_cond)

    for _ in range(max_new_tokens):
        logits_last = logits[:, -1, :]
        probs = F.softmax(logits_last, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        logits, _, past_kvs = model(idx_next, past_kvs=past_kvs)

    return idx


def main(cfg=None):
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)

    tokenizer, train_data, val_data = load_data(cfg)
    cfg.vocab_size = tokenizer.vocab_size
    model = GPT(cfg).to(cfg.device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{num_params:.3f} M parameters")

    train(model, train_data, val_data, cfg)

    start = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    generated = generate(model, start, 1000, cfg)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()