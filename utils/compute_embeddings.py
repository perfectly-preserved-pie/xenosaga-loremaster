#!/usr/bin/env python3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_glossary(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_documents(glossary: list) -> list:
    documents = []
    for entry in glossary:
        doc = (
            f"Term: {entry.get('term', '')}\n"
            f"Definition: {entry.get('definition', '')}\n"
            f"Source: {entry.get('source', '')}\n"
            f"Episode: {entry.get('episode', '')}\n"
            f"URL: {entry.get('source_url', '')}"
        )
        documents.append(doc)
    return documents

def compute_embeddings(documents: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    # Load the model on GPU
    model = SentenceTransformer(model_name, device="cuda")
    print(f"Using model: {model_name} on {model.device}")
    
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    return embeddings

def main():
    json_path = "corpus/output/glossary.json"
    glossary = load_glossary(json_path)
    documents = prepare_documents(glossary)
    embeddings = compute_embeddings(documents)
    
    print(f"Computed embeddings for {len(documents)} documents.")
    print("Embedding shape:", embeddings.shape)
    
    np.save("corpus/output/glossary_embeddings.npy", embeddings)
    print("Embeddings saved to 'glossary_embeddings.npy'.")

if __name__ == "__main__":
    main()
