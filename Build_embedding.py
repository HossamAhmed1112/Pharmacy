import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading products...")

df = pd.read_excel("Pharmacy_Products_ALL_added.xlsx")


def build_search_text(row):
    fields = []

    for column in [
        "name",
        "name_ar",
        "alternate_names",
        "synonyms",
        "packaging"
    ]:
        if column in row and pd.notna(row[column]):
            fields.append(str(row[column]))

    return " ".join(fields).strip()


search_texts = [
    build_search_text(row)
    for _, row in df.iterrows()
]

print(f"Encoding {len(search_texts)} products...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)
embeddings = model.encode(
    search_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save("embeddings.npy", embeddings)

print("Embeddings saved successfully")