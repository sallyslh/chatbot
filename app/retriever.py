#after normalization takes the user question and retrieves the most relevant chunks from the index this is for the blnk data
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.normalize import normalize_query

INDEX_PATH = "data/blnk.index"
META_PATH = "data/blnk_meta.json"

model = SentenceTransformer("BAAI/bge-m3")

_index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    _meta = json.load(f)

def retrieve(query: str, top_k: int = 5, use_normalization: bool = True):
    search_text = normalize_query(query) if use_normalization else query

    vec = model.encode([search_text], normalize_embeddings=True)
    vec = np.array(vec, dtype="float32")

    scores, idxs = _index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = _meta[idx]
        results.append({"score": float(score), **chunk})
    return results