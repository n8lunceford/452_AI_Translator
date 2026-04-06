"""
app.py

Flask web server for the English → Spanish translator.
Run with: python app.py
Then open: http://localhost:5000
"""

import math
import pickle
import datetime
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from bson import ObjectId
from pymongo import MongoClient

# ─────────────────────────── Config ─────────────────────────────────────────

MODEL_SAVE  = "model.pt"
TOK_SAVE    = "tokenizer.pkl"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
MAX_DECODE_LEN = 128

MONGO_URI  = "mongodb://localhost:27017/"
MONGO_DB   = "translations_db"
MONGO_COL  = "saved_translations"

# ─────────────────────────── Model Architecture ──────────────────────────────

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
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len):
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
        memory  = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding)
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
        return self.fc_out(out)


# ─────────────────────────── Load Model & Tokenizer ──────────────────────────

def fix_encoding(text):
    text = text.replace("Ã¡", "á")
    text = text.replace("Ã©", "é")
    text = text.replace("Ã­", "í")
    text = text.replace("Ã³", "ó")
    text = text.replace("Ãº", "ú")
    text = text.replace("Ã±", "ñ")
    text = text.replace("Ã¼", "ü")
    text = text.replace("Â¿", "¿")
    text = text.replace("Â¡", "¡")
    text = text.replace("Ã", "Á")
    text = text.replace("Ã‰", "É")
    text = text.replace("Ã\x93", "Ó")
    text = text.replace("Ãš", "Ú")
    text = text.replace("Ã'", "Ñ")
    text = text.replace("Ġ", " ")
    return text.strip()


print("[info] Loading tokenizer and model …")
with open(TOK_SAVE, "rb") as f:
    tokenizer = pickle.load(f)

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
print(f"[info] Model loaded. Device: {DEVICE}")


# ─────────────────────────── MongoDB ─────────────────────────────────────────

def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client[MONGO_DB][MONGO_COL]


# ─────────────────────────── Translation ─────────────────────────────────────

def greedy_translate(sentence):
    with torch.no_grad():
        enc = tokenizer.encode(sentence)
        src_ids = torch.tensor([enc.ids], dtype=torch.long, device=DEVICE)
        memory, mem_pad_mask = model.encode(src_ids)
        tgt_ids = torch.tensor([[BOS_IDX]], dtype=torch.long, device=DEVICE)
        for _ in range(MAX_DECODE_LEN):
            logits = model.decode_step(tgt_ids, memory, mem_pad_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_token.item() == EOS_IDX:
                break
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        output_ids = tgt_ids[0, 1:].tolist()
        translation = tokenizer.decode(output_ids, skip_special_tokens=True)
        return fix_encoding(translation)


# ─────────────────────────── Flask App ───────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    english = data.get("text", "").strip()
    if not english:
        return jsonify({"error": "No input provided."}), 400
    spanish = greedy_translate(english)
    return jsonify({"english": english, "spanish": spanish})


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    english = data.get("english", "").strip()
    spanish = data.get("spanish", "").strip()
    if not english or not spanish:
        return jsonify({"error": "Missing fields."}), 400
    col = get_collection()
    result = col.insert_one({
        "english": english,
        "spanish": spanish,
        "timestamp": datetime.datetime.utcnow(),
    })
    return jsonify({"message": "Saved!", "id": str(result.inserted_id)})


@app.route("/translations", methods=["GET"])
def get_translations():
    col = get_collection()
    docs = list(col.find().sort("timestamp", -1))
    result = []
    for doc in docs:
        result.append({
            "id": str(doc["_id"]),
            "english": doc["english"],
            "spanish": doc["spanish"],
            "timestamp": doc["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        })
    return jsonify(result)


@app.route("/delete/<doc_id>", methods=["DELETE"])
def delete_translation(doc_id):
    col = get_collection()
    result = col.delete_one({"_id": ObjectId(doc_id)})
    if result.deleted_count == 1:
        return jsonify({"message": "Deleted."})
    return jsonify({"error": "Not found."}), 404


@app.route("/clear", methods=["DELETE"])
def clear_translations():
    col = get_collection()
    result = col.delete_many({})
    return jsonify({"message": f"Cleared {result.deleted_count} document(s)."})


if __name__ == "__main__":
    app.run(debug=True)
