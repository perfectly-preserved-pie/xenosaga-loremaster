#!/usr/bin/env python3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_glossary(json_path: str) -> list:
    """
    Load glossary entries from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file.
    
    Returns:
        list: A list of glossary entry dictionaries.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_documents(glossary: list) -> list:
    """
    Combine fields from each glossary entry into a single string.
    
    Args:
        glossary (list): List of glossary entry dictionaries.
    
    Returns:
        list: A list of strings, one per entry.
    """
    documents = []
    for entry in glossary:
        # Combine the term, definition, source, and episode into one text block.
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
    """
    Compute embeddings for a list of documents using a Sentence Transformer.
    
    Args:
        documents (list): A list of text strings.
        model_name (str): The model name to load from Sentence Transformers.
    
    Returns:
        np.ndarray: An array of embeddings (float32).
    """
    # Load the pre-trained sentence transformer model.
    model = SentenceTransformer(model_name)
    print(f"Using model: {model_name}")
    
    # Compute embeddings for each document.
    embeddings = model.encode(documents, show_progress_bar=True)
    
    # Convert to a NumPy array of type float32 (required by many vector search libraries like FAISS).
    embeddings = np.array(embeddings).astype("float32")
    return embeddings

def main():
    # Specify the path to your JSON glossary file.
    json_path = "corpus/output/glossary_entries.json"
    
    # Load and prepare the glossary data.
    glossary = load_glossary(json_path)
    documents = prepare_documents(glossary)
    
    # Compute embeddings.
    embeddings = compute_embeddings(documents)
    
    # Print some information about the embeddings.
    print(f"Computed embeddings for {len(documents)} documents.")
    print("Embedding shape:", embeddings.shape)
    
    # Optionally, save the embeddings for later use.
    np.save("corpus/output/glossary_embeddings.npy", embeddings)
    print("Embeddings saved to 'glossary_embeddings.npy'.")

if __name__ == "__main__":
    main()
