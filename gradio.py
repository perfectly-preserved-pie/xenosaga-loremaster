import gradio as gr
from utils.build_faiss import retrieve_entries
from utils.compute_embeddings import generate_answer

def rag_pipeline(query: str) -> str:
    """
    A simple RAG pipeline that:
    1. Retrieves relevant glossary entries for the query.
    2. Constructs a prompt and generates an answer using a generation model.
    
    Args:
        query (str): The user query.
    
    Returns:
        str: The generated answer.
    """
    # Retrieve the top 3 glossary entries.
    retrieved_docs, distances = retrieve_entries(query, top_k=3)
    
    # Generate an answer using the retrieved context.
    answer = generate_answer(query, retrieved_docs, max_length=256)
    return answer

# Create a Gradio interface.
interface = gr.Interface(
    fn=rag_pipeline,
    inputs=gr.inputs.Textbox(label="Enter your question", placeholder="What does AAA class encryption mean?"),
    outputs=gr.outputs.Textbox(label="Generated Answer"),
    title="RAG System for Xenosaga Glossary",
    description="Enter a question about the glossary and see the answer generated using retrieved context."
)

# Launch the interface.
interface.launch()
