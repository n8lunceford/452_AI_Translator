"""
train_translator.py

Trains a Transformer model from scratch on an English→Spanish translation
dataset stored in a CSV file with two columns: 'english' and 'spanish'
(column names are auto-detected — see load_data()).

After training the script saves:
  - model.pt          : model weights + config
  - tokenizer.pkl     : fitted BPE tokenizer (src & tgt vocabs)

Usage:
    python train_translator.py

Tested with Python 3.10+ and PyTorch 2.x.
"""

import os
import math
import time
import pickle
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# ─────────────────────────────── Config ────────────────────────────────────

CSV_PATH      = "en_sp_translations.csv"   # path to your CSV
MODEL_SAVE    = "model.pt"
TOK_SAVE      = "tokenizer.pkl"

VOCAB_SIZE    = 16000      # shared BPE vocabulary size
MAX_LEN       = 128        # max tokens per sentence
D_MODEL       = 256        # transformer embedding dim
N_HEADS       = 8          # attention heads
N_LAYERS      = 3          # encoder & decoder layers each
D_FF          = 512        # feed-forward hidden dim
DROPOUT       = 0.1
BATCH_SIZE    = 128
EPOCHS        = 10
LR            = 3e-4
WARMUP_STEPS  = 4000
CLIP          = 1.0        # gradient clipping
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ───────────────────────────── Data Loading ─────────────────────────────────

def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Try common column name variants
    eng_candidates = ["english", "en", "source", "src"]
    esp_candidates = ["spanish", "es", "sp", "target", "tgt"]

    eng_col = next((c for c in eng_candidates if c in df.columns), None)
    esp_col = next((c for c in esp_candidates if c in df.columns), None)

    if eng_col is None or esp_col is None:
        # Fall back to positional (first two columns)
        print(f"[warn] Could not auto-detect column names. Columns found: {list(df.columns)}")
        print("[warn] Assuming first column = English, second = Spanish")
        eng_col, esp_col = df.columns[0], df.columns[1]
    else:
        print(f"[info] Detected columns: english='{eng_col}', spanish='{esp_col}'")

    pairs = df[[eng_col, esp_col]].dropna().astype(str).values.tolist()
    random.shuffle(pairs)
    return pairs


# ─────────────────────────── BPE Tokenizer ──────────────────────────────────

def train_tokenizer(pairs, vocab_size=VOCAB_SIZE):
    """Train a shared BPE tokenizer on both languages."""
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        min_frequency=2,
    )

    # Feed all sentences to the trainer
    all_sentences = [en for en, _ in pairs] + [es for _, es in pairs]
    tokenizer.train_from_iterator(all_sentences, trainer=trainer)

    # Add post-processor so [BOS]/[EOS] are prepended/appended automatically
    bos = tokenizer.token_to_id("[BOS]")
    eos = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", bos), ("[EOS]", eos)],
    )
    tokenizer.enable_padding(pad_id=PAD_IDX, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=MAX_LEN)
    return tokenizer


# ───────────────────────────────── Dataset ───────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=MAX_LEN):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, es = self.pairs[idx]
        enc_en = self.tokenizer.encode(en)
        enc_es = self.tokenizer.encode(es)

        src_ids = enc_en.ids[: self.max_len]
        tgt_ids = enc_es.ids[: self.max_len]

        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [s.size(0) for s in src_batch]
    tgt_lens = [t.size(0) for t in tgt_batch]
    max_src  = max(src_lens)
    max_tgt  = max(tgt_lens)

    src_padded = torch.full((len(src_batch), max_src), PAD_IDX, dtype=torch.long)
    tgt_padded = torch.full((len(tgt_batch), max_tgt), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, : s.size(0)] = s
        tgt_padded[i, : t.size(0)] = t

    return src_padded, tgt_padded


# ─────────────────────────── Transformer Model ──────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc   = PositionalEncoding(d_model, dropout, max_len + 2)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, src, tgt):
        src_key_padding = (src == PAD_IDX)
        tgt_key_padding = (tgt == PAD_IDX)
        tgt_mask        = self.make_causal_mask(tgt.size(1), tgt.device)

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=src_key_padding,
        )
        return self.fc_out(out)   # (batch, tgt_len, vocab_size)


# ─────────────────────────── LR Scheduler ───────────────────────────────────

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps):
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self._step        = 0
        super().__init__(optimizer)

    def get_lr(self):
        self._step += 1
        scale = self.d_model ** -0.5 * min(
            self._step ** -0.5,
            self._step * self.warmup_steps ** -1.5
        )
        return [scale for _ in self.base_lrs]


# ──────────────────────────── Training Loop ─────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for step, (src, tgt) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in  = tgt[:, :-1]   # decoder input  (drop last token)
        tgt_out = tgt[:, 1:]    # decoder target (drop first [BOS])

        logits = model(src, tgt_in)   # (B, T-1, V)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_out = tgt_out.reshape(-1)

        loss = criterion(logits, tgt_out)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if (step + 1) % 200 == 0:
            print(f"  step {step+1}/{len(loader)}  loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
    return total_loss / len(loader)


# ──────────────────────────────── Main ──────────────────────────────────────

def main():
    print(f"[info] Using device: {DEVICE}")

    # 1. Load data
    print("[info] Loading data …")
    pairs = load_data(CSV_PATH)
    print(f"[info] {len(pairs)} sentence pairs loaded.")

    # 2. Train tokenizer
    print("[info] Training BPE tokenizer …")
    tokenizer = train_tokenizer(pairs, VOCAB_SIZE)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[info] Vocabulary size: {vocab_size}")

    # 3. Save tokenizer
    with open(TOK_SAVE, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"[info] Tokenizer saved → {TOK_SAVE}")

    # 4. Dataset / DataLoader
    split = int(0.95 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    train_ds = TranslationDataset(train_pairs, tokenizer)
    val_ds   = TranslationDataset(val_pairs,   tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # 5. Model
    model = Seq2SeqTransformer(vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] Model parameters: {n_params:,}")

    # 6. Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, D_MODEL, WARMUP_STEPS)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    # 7. Training
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, criterion, DEVICE)
        print(f"[epoch {epoch}/{EPOCHS}] train_loss={train_loss:.4f}  time={time.time()-t0:.0f}s")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                logits = model(src, tgt[:, :-1])
                logits = logits.reshape(-1, logits.size(-1))
                val_loss += criterion(logits, tgt[:, 1:].reshape(-1)).item()
        val_loss /= len(val_dl)
        print(f"[epoch {epoch}/{EPOCHS}] val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": D_MODEL,
                    "n_heads": N_HEADS,
                    "n_layers": N_LAYERS,
                    "d_ff": D_FF,
                    "dropout": DROPOUT,
                    "max_len": MAX_LEN,
                },
            }, MODEL_SAVE)
            print(f"[info] Model saved → {MODEL_SAVE}  (best val_loss={best_val:.4f})")

    print("[done] Training complete.")


if __name__ == "__main__":
    main()
