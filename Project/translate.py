"""
translate.py

Interactive English → Spanish translator using the model trained by
train_translator.py.

Usage:
    python translate.py

Commands at the prompt:
    <any English text>  →  translate it
    1                   →  save the last translation to MongoDB
    0                   →  quit

Requirements (install once):
    pip install torch tokenizers pymongo

MongoDB must be running locally on the default port (27017).
See README / setup notes at the bottom of this file if you haven't
installed MongoDB yet.
"""

import math
import pickle
import sys
import datetime

import torch
import torch.nn as nn

# ───────────────────────── Must match train_translator.py ───────────────────

MODEL_SAVE  = "model.pt"
TOK_SAVE    = "tokenizer.pkl"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

MAX_DECODE_LEN = 128

# MongoDB settings
MONGO_URI  = "mongodb://localhost:27017/"
MONGO_DB   = "translations_db"
MONGO_COL  = "saved_translations"


# ─────────────────── Transformer (identical architecture) ───────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 dropout, max_len):
        super().__init__()
        self.d_model   = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc   = PositionalEncoding(d_model, dropout, max_len + 2)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=d_ff, dropout=dropout, batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        src_key_padding = (src == PAD_IDX)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory  = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding)
        return memory, src_key_padding

    def decode_step(self, tgt, memory, memory_key_padding_mask):
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), device=tgt.device), diagonal=1
        ).bool()
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(out)   # (1, T, V)


# ───────────────────────────── Greedy decode ────────────────────────────────

def greedy_translate(model, tokenizer, sentence: str,
                     device=DEVICE, max_len=MAX_DECODE_LEN) -> str:
    model.eval()
    with torch.no_grad():
        # Encode source
        enc = tokenizer.encode(sentence)
        src_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)

        memory, mem_pad_mask = model.encode(src_ids)

        # Decoder starts with [BOS]
        tgt_ids = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = model.decode_step(tgt_ids, memory, mem_pad_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1,1)
            if next_token.item() == EOS_IDX:
                break
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

        # Decode output tokens (skip [BOS])
        output_ids = tgt_ids[0, 1:].tolist()
        translation = tokenizer.decode(output_ids, skip_special_tokens=True)
        # translation = bytearray([ord(c) for c in translation if ord(c) < 256]).decode("utf-8", errors="ignore") # UTF-8 fix
        # translation = translation.replace("Ġ", " ").strip()                                                     # UTF-8 fix
        # translation = translation.replace("ÁŃ", "í").strip()
        translation = translation.replace("Ã¡", "á").strip()
        translation = translation.replace("Ã©", "é").strip()
        translation = translation.replace("Ã­", "í").strip()
        translation = translation.replace("Ã³", "ó").strip()
        translation = translation.replace("Ãº", "ú").strip()
        translation = translation.replace("Ã±", "ñ").strip()
        translation = translation.replace("Ã¼", "ü").strip()
        translation = translation.replace("Â¿", "¿").strip()
        translation = translation.replace("Â¡", "¡").strip()
        translation = translation.replace("Ã", "Á").strip()
        translation = translation.replace("Ã‰", "É").strip()
        translation = translation.replace("Ã", "Í").strip()
        translation = translation.replace("Ã\x93", "Ó").strip()
        translation = translation.replace("Ãš", "Ú").strip()
        translation = translation.replace("Ã'", "Ñ").strip()
        translation = translation.replace("Ġ", " ").strip()
        return translation


# ─────────────────────────── MongoDB helpers ────────────────────────────────

def get_mongo_collection():
    """Return the MongoDB collection, or None if unavailable."""
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()               # raises if not reachable
        return client[MONGO_DB][MONGO_COL]
    except Exception as e:
        print(f"[MongoDB error] {e}")
        return None


def save_to_mongo(collection, english: str, spanish: str):
    doc = {
        "english":   english,
        "spanish":   spanish,
        "timestamp": datetime.datetime.utcnow(),
    }
    result = collection.insert_one(doc)
    print(f"[saved] Document inserted with id: {result.inserted_id}")


# ───────────────────────────────── Main ─────────────────────────────────────

def load_model_and_tokenizer():
    print("[info] Loading tokenizer …")
    with open(TOK_SAVE, "rb") as f:
        tokenizer = pickle.load(f)

    print("[info] Loading model …")
    checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE)
    cfg = checkpoint["config"]
    model = Seq2SeqTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def main():
    # ── load model ──────────────────────────────────────────────────────────
    try:
        model, tokenizer = load_model_and_tokenizer()
    except FileNotFoundError as e:
        print(f"[error] {e}")
        print("Make sure you have run train_translator.py first.")
        sys.exit(1)

    print(f"[info] Using device: {DEVICE}")
    print()
    print("=" * 60)
    print("  English → Spanish Translator")
    print("  Type an English sentence to translate.")
    print("  Type  1  to save the last translation to MongoDB.")
    print("  Type  0  to quit.")
    print("=" * 60)

    # ── Lazy MongoDB connection (only when the user presses 1) ───────────────
    mongo_col        = None
    last_english     = None
    last_spanish     = None

    while True:
        try:
            user_input = input("\nEnglish > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not user_input:
            continue

        # ── Quit ────────────────────────────────────────────────────────────
        if user_input == "0":
            print("[bye]")
            break

        # ── Clear MongoDB table ──────────────────────────────────────────────
        if user_input == "4":
            if mongo_col is None:
                print("[info] Connecting to MongoDB …")
                mongo_col = get_mongo_collection()

            if mongo_col is not None:
                result = mongo_col.delete_many({})
                print(f"[cleared] {result.deleted_count} document(s) removed from the collection.")
            else:
                print("[error] Could not connect to MongoDB. Is it running?")
            continue
        
        # ── Display all saved translations ───────────────────────────────────
        if user_input == "2":
            if mongo_col is None:
                print("[info] Connecting to MongoDB …")
                mongo_col = get_mongo_collection()

            if mongo_col is not None:
                docs = list(mongo_col.find())
                if not docs:
                    print("[info] No translations saved yet.")
                else:
                    print(f"\n{'─' * 60}")
                    for doc in docs:
                        print(f"  ID      : {doc['_id']}")
                        print(f"  English : {doc['english']}")
                        print(f"  Spanish : {doc['spanish']}")
                        print(f"  Saved   : {doc['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"{'─' * 60}")
            else:
                print("[error] Could not connect to MongoDB. Is it running?")
            continue

        # ── Delete a translation by ID ───────────────────────────────────────
        if user_input == "3":
            if mongo_col is None:
                print("[info] Connecting to MongoDB …")
                mongo_col = get_mongo_collection()

            if mongo_col is not None:
                docs = list(mongo_col.find())
                if not docs:
                    print("[info] No translations saved yet.")
                else:
                    print(f"\n{'─' * 60}")
                    for doc in docs:
                        print(f"  ID      : {doc['_id']}")
                        print(f"  English : {doc['english']}")
                        print(f"  Spanish : {doc['spanish']}")
                        print(f"{'─' * 60}")
                    target_id = input("Enter the ID to delete > ").strip()
                    try:
                        from bson import ObjectId
                        result = mongo_col.delete_one({"_id": ObjectId(target_id)})
                        if result.deleted_count == 1:
                            print(f"[deleted] Translation {target_id} removed.")
                        else:
                            print(f"[warn] No translation found with that ID.")
                    except Exception as e:
                        print(f"[error] Invalid ID format: {e}")
            else:
                print("[error] Could not connect to MongoDB. Is it running?")
            continue

        # ── Save last translation ────────────────────────────────────────────
        if user_input == "1":
            if last_english is None:
                print("[warn] No translation to save yet. Translate something first.")
                continue

            if mongo_col is None:
                print("[info] Connecting to MongoDB …")
                mongo_col = get_mongo_collection()

            if mongo_col is not None:
                save_to_mongo(mongo_col, last_english, last_spanish)
            else:
                print("[error] Could not connect to MongoDB. Is it running?")
                print("        See the setup notes at the bottom of translate.py.")
            continue

        # ── Translate ────────────────────────────────────────────────────────
        translation = greedy_translate(model, tokenizer, user_input)
        print(f"Spanish > {translation}")
        last_english = user_input
        last_spanish = translation


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# MONGODB SETUP (if you haven't installed it yet)
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Install MongoDB Community Edition
#    https://www.mongodb.com/try/download/community
#    (no account required for the Community edition)
#
#    macOS (Homebrew):
#        brew tap mongodb/brew
#        brew install mongodb-community
#        brew services start mongodb-community
#
#    Ubuntu / Debian:
#        sudo apt-get install -y mongodb
#        sudo systemctl start mongodb
#
#    Windows:
#        Download the MSI installer from the link above.
#        The installer can set it up as a Windows Service automatically.
#
# 2. Install the Python driver (once):
#        pip install pymongo
#
# 3. That's it — no account, no sign-up needed for a local instance.
#    Data is stored in the "translations_db" database,
#    "saved_translations" collection.
#
# 4. To browse saved translations:
#        mongosh
#        use translations_db
#        db.saved_translations.find().pretty()
# ─────────────────────────────────────────────────────────────────────────────
