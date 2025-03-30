import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import tiktoken

# Configuration
class Config:
    vocab_size = 50257  # GPT-2 tokenizer vocab size
    n_embd = 384        # embedding dimension
    n_head = 6          # number of attention heads
    n_layer = 6         # number of transformer layers
    block_size = 256    # context window length
    dropout = 0.1       # dropout rate
    lr = 3e-4           # learning rate
    batch_size = 32     # training batch size

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Model Components
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (x.size(-1)**-0.5)
        wei = wei.masked_fill(self.tril[:wei.size(1), :wei.size(2)] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd // n_head, n_embd, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(Config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(Config.vocab_size, Config.n_embd)
        self.position_embedding = nn.Embedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[Block(Config.n_embd, Config.n_head, Config.block_size) for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, Config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -Config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class MovieScriptDataset(Dataset):
    def __init__(self, dataset, tokenizer, block_size):
        self.examples = []
        for movie in dataset:
            script = self._preprocess_script(movie['Script'])
            tokens = tokenizer.encode(script)
            if len(tokens) >= block_size:
                self.examples.extend([tokens[i:i + block_size + 1] for i in range(0, len(tokens) - block_size, block_size // 2)])

    def _preprocess_script(self, script):
        lines = [line.strip() for line in script.split('\n') if line.strip()]
        cleaned = []
        for line in lines:
            if line.isupper() and len(line) < 30:
                cleaned.append(f"\n{line}: ")
            else:
                cleaned.append(line)
        return " ".join(cleaned)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Loading movie scripts dataset...")
    movies = load_dataset("IsmaelMousa/movies", split="train")
    dataset = MovieScriptDataset(movies, tokenizer, Config.block_size)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    model = GPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)

    model.train()
    for epoch in range(1):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "movie_gpt.pth")
    return model

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = GPT().to(device)
    model.load_state_dict(torch.load("movie_gpt.pth", map_location=device))
    return model

def generate_script(model, prompt):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt)
    context = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"Generating script starting with: {prompt}")
    generated = model.generate(context, max_new_tokens=500, temperature=0.8)
    print(tokenizer.decode(generated[0].tolist()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py '<prompt>'")
        sys.exit(1)

    prompt = sys.argv[1]

    if os.path.exists("movie_gpt.pth"):
        print("Loading the saved model...")
        model = load_model()
    else:
        print("Trained model not found, training a new model...")
        model = train_model()

    generate_script(model, prompt)
