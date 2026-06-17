import os
from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Model loaded")


def embed_texts(texts):
    print(f"Encoding {len(texts)} texts...")
    result = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print("Encoding finished")
    return result


def embed_query(query):
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]


try:
    from annoy import AnnoyIndex
    _has_annoy = True
except Exception:
    _has_annoy = False


def build_annoy_index(embeddings, path="embeddings.ann", n_trees=10):
    if not _has_annoy:
        return False

    dim = embeddings.shape[1]
    t = AnnoyIndex(dim, "angular")

    for i, v in enumerate(embeddings):
        t.add_item(i, v.tolist())

    t.build(n_trees)
    t.save(path)
    return True


def load_annoy_index(dim, path="embeddings.ann"):
    if not _has_annoy:
        return None

    if not os.path.exists(path):
        return None

    t = AnnoyIndex(dim, "angular")
    t.load(path)
    return t