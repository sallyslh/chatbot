#vectorize the chunks and build the index
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/blnk_chunks.jsonl"
INDEX_PATH = "data/blnk.index"
META_PATH = "data/blnk_meta.json"

model = SentenceTransformer("BAAI/bge-m3")

def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def build_index():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    vectors = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Indexed {len(chunks)} chunks.")

if __name__ == "__main__":
    build_index()