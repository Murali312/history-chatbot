import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Load env variables immediately
load_dotenv()

class HistoryBotBackend:
    def __init__(self, pdf_path="historical_figures.pdf"):
        self.pdf_path = pdf_path
        self.chat_history = InMemoryChatMessageHistory()
        self.rag_chain = self._initialize_system()

    def _load_and_process_pdf(self):
        if not os.path.exists(self.pdf_path):
            print(f"Error: {self.pdf_path} not found.")
            return None
        print("Loading PDF...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        print("Splitting text...")
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        splits = text_splitter.split_documents(documents)
        return splits

    def _setup_vector_store(self, splits):
        if not splits: return None
        print("Initializing Vector Store...")
        embeddings = OllamaEmbeddings(model="granite-embedding:latest")
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="historical_figures"
        )
        return vectorstore

    def _setup_rag_pipeline(self, vectorstore):
        if not vectorstore: return None
        print("Setting up Llama3...")
        llm = ChatOllama(model="llama3.2:3b")
        
        template = """You are HistoryBot, an expert on historical figures. 
        Use the following context to answer the question.
        
        Context: {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        retriever = vectorstore.as_retriever()

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    def _initialize_system(self):
        """Orchestrates the startup process."""
        splits = self._load_and_process_pdf()
        vectorstore = self._setup_vector_store(splits)
        return self._setup_rag_pipeline(vectorstore)

    def generate_response(self, user_input):
        """Processes input and updates history."""
        if not self.rag_chain:
            return "System Error: PDF loading failed or system uninitialized."

        # 1. Update internal history with user query
        self.chat_history.add_message(HumanMessage(content=user_input))
        
        # 2. Invoke Chain
        try:
            answer = self.rag_chain.invoke(user_input)
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        # 3. Update internal history with AI response
        self.chat_history.add_message(AIMessage(content=answer))
        
        return answer

    def get_gradio_history(self):
        """Formats the internal history specifically for the Gradio UI."""
        # Returns: [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]
        gradio_format = []
        for msg in self.chat_history.messages:
            if isinstance(msg, HumanMessage):
                gradio_format.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                gradio_format.append({"role": "assistant", "content": msg.content})
        return gradio_format

    def clear_memory(self):
        self.chat_history.clear()
        return []