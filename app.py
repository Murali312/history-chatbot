import gradio as gr
from rag_backend import HistoryBotBackend

# --- Initialize Backend ---
# This starts the PDF loading and Vector Store creation immediately
backend = HistoryBotBackend(pdf_path="historical_figures.pdf")

# --- UI Event Functions ---
def chat_wrapper(user_input, history_state):
    """
    Wrapper to connect Gradio input to Backend logic.
    Returns: (Empty String for textbox, Updated History list)
    """
    # 1. Generate the response (this updates backend state)
    backend.generate_response(user_input)
    
    # 2. Retrieve the formatted history for display
    # We retrieve the dictionary format [{'role': 'user', ...}]
    updated_history = backend.get_gradio_history()
    
    return "", updated_history

def clear_wrapper():
    """Wrapper to clear backend memory and UI."""
    return backend.clear_memory()

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("### Hello, I am Historybot. Your expert on historical figures.")
    
    # FIX: Removed 'type="messages"' to prevent the TypeError crash.
    # The backend will still send dictionary data, which the UI should accept.
    chatbot = gr.Chatbot(label="Conversation", height=400)
    msg = gr.Textbox(label="Your Question", placeholder="Ask about historical figures...")
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear Chat History")

    # Event Wiring
    # Note: We pass the 'backend' methods via our wrapper functions
    submit_btn.click(chat_wrapper, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat_wrapper, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(clear_wrapper, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()