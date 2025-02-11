import gradio as gr
from utils.build_faiss import retrieve_entries

def query_docs(query: str) -> str:
    retrieved_docs, distances = retrieve_entries(query, top_k=3)
    results = []
    for i, (doc, dist) in enumerate(zip(retrieved_docs, distances[0])):
        results.append(f"Result {i + 1} (Distance: {dist:.2f}):\n{doc}\n")
    return "\n".join(results)

iface = gr.Interface(
    fn=query_docs,
    inputs=gr.Textbox(lines=2, label="Enter Query"),
    outputs=gr.Textbox(label="Retrieved Documents"),
    title="Xenosaga Glossary Query Interface",
    description="Enter a query to retrieve relevant glossary entries."
)

if __name__ == "__main__":
    iface.launch()