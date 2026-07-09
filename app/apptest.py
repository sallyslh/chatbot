import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/blnk.index"
META_PATH = "data/blnk_meta.json"

model = SentenceTransformer("BAAI/bge-m3")

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

def search(query, top_k=5):
    vec = model.encode([query], normalize_embeddings=True)
    vec = np.array(vec, dtype="float32")
    scores, idxs = index.search(vec, top_k)

    print(f"\nQuery: {query}")
    print("-" * 60)
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = meta[idx]
        print(f"Score: {score:.4f} | id: {chunk['id']}")
        print(f"  {chunk['text'][:150]}...")
    print()

if __name__ == "__main__":
    test_queries = [
        "do you have real estate clients in Qatar?",
        "what is your design process?",
        "do you do luxury branding?",
        "can I get pricing?",
        "who owns the agency?",
        "do you work in Arabic?",
        "شلون أقدر أعرف الأسعار عندكم؟",
    ]

    for q in test_queries:
        search(q)