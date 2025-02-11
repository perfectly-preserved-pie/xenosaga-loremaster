#!/usr/bin/env python3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Build Phase ---

# Load your embeddings
embeddings = np.load("corpus/output/glossary_embeddings.npy")
print("Embeddings shape:", embeddings.shape)  # e.g., (num_entries, embedding_dimension)
embedding_dim = embeddings.shape[1]

# Build a CPU index first (a simple L2 index)
cpu_index = faiss.IndexFlatL2(embedding_dim)
cpu_index.add(embeddings)
print(f"Number of entries indexed (CPU): {cpu_index.ntotal}")

# Transfer the index to the GPU using device 0 of the first available GPU
gpu_res = faiss.StandardGpuResources()  # initialize GPU resources
gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
print("GPU index built and transferred.")

# Save the GPU index to disk
faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), "glossary_index.faiss")
print("GPU index saved to 'glossary_index.faiss'.")

# --- Query Phase ---

# Load the original glossary JSON to retrieve the corresponding documents.
with open("corpus/output/glossary.json", "r", encoding="utf-8") as f:
    glossary = json.load(f)

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

# Load the same Sentence Transformer model on GPU for query embeddings
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

def retrieve_entries(query: str, top_k: int = 3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    
    distances, indices = gpu_index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs, distances[0]

# Example query:
query = "What does AAA class encryption mean?"
results, distances = retrieve_entries(query, top_k=3)

print("\nRetrieved Glossary Entries:")
for i, (doc, dist) in enumerate(zip(results, distances), 1):
    print(f"\nEntry {i} (Distance: {dist:.4f}):\n{doc}")
