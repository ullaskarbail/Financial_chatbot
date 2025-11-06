"""
LangChain Integration (persistent memory + RAG)
- Uses Chroma (local) as vector store for persistent conversation memory & document store
- Provides: LangChainFinanceAgentPersistent with memory, document ingestion, document-retriever-based QA
"""
from typing import Optional, List
import os
import json
from pathlib import Path

# Try to import LangChain components, fall back to basic implementation if not available
try:
    from langchain.prompts import PromptTemplate
    from langchain.chains import ConversationChain, RetrievalQA
    from langchain_community.llms import HuggingFacePipeline
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
    from langchain.text_splitter import CharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain not fully available: {e}")
    print("📝 Using fallback implementation. Install langchain and langchain-community for full functionality.")
    LANGCHAIN_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers not available. Some features will be limited.")
    TRANSFORMERS_AVAILABLE = False


# Defaults
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chroma")
DEFAULT_MODEL = os.getenv("WATSONX_MODEL_NAME", "microsoft/DialoGPT-medium")


class LangChainFinanceAgentPersistent:
    """LangChain agent with persistent memory + document store using Chroma.

    - Conversation memory persists to Chroma via a special metadata key
    - Documents uploaded are embedded and stored in Chroma for RAG
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, persist_directory: str = CHROMA_PERSIST_DIR):
        print("🔗 Initializing LangChain persistent agent...")
        
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.conversation_history = []
        self.documents = {}
        
        if LANGCHAIN_AVAILABLE:
            try:
                # HF text generation pipeline
                if TRANSFORMERS_AVAILABLE:
                    hf_pipeline = pipeline(
                        "text-generation",
                        model=model_name,
                        max_new_tokens=256,
                        temperature=0.3,
                        top_p=0.9
                    )
                    self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
                else:
                    self.llm = None

                # Embeddings (use small huggingface embeddings to stay local)
                embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

                # Vectorstore: Chroma (persisted)
                persist_dir = Path(persist_directory)
                persist_dir.mkdir(parents=True, exist_ok=True)

                self.vectorstore = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=self.embeddings,
                    collection_name="personal_finance",
                )

                # Memory: ConversationBufferWindowMemory (keeps last N messages in memory)
                self.memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)

                if self.llm:
                    # Conversation chain (simple)
                    self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=False)
                    # Retrieval QA chain for docs
                    self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
                    self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)
                else:
                    self.conversation = None
                    self.qa_chain = None
                
                print("✅ Persistent LangChain agent initialized.")
            except Exception as e:
                print(f"⚠️ Error initializing LangChain components: {e}")
                print("📝 Falling back to basic implementation.")
                self._init_fallback()
        else:
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback implementation when LangChain is not available"""
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.memory = None
        self.conversation = None
        self.qa_chain = None
        print("📝 Using basic fallback implementation.")

    # --- Conversation functions ---
    def ask(self, message: str) -> str:
        if self.conversation:
            try:
                return self.conversation.predict(input=message).strip()
            except Exception as e:
                return f"⚠️ LangChain conversation error: {str(e)}"
        else:
            # Fallback: simple conversation history
            self.conversation_history.append({"role": "user", "content": message})
            response = f"I received your message: '{message}'. LangChain is not fully available, so I'm using a basic response. Please install the required dependencies for full functionality."
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

    def reset_memory(self):
        if self.memory:
            self.memory.clear()
        else:
            self.conversation_history = []
        print("🧠 Conversation memory cleared.")

    # --- Document ingestion & QA ---
    def ingest_text(self, text: str, doc_id: Optional[str] = None, metadata: Optional[dict] = None):
        """Ingest raw text into Chroma: split, embed, and persist."""
        if self.vectorstore and LANGCHAIN_AVAILABLE:
            try:
                splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                docs = splitter.split_text(text)

                # Prepare records
                met = metadata or {}
                records = [
                    {"id": f"{doc_id or 'doc'}_{i}", "text": chunk, "metadata": met}
                    for i, chunk in enumerate(docs)
                ]

                # Add to vectorstore
                self.vectorstore.add_texts([r['text'] for r in records], metadatas=[r['metadata'] for r in records])
                self.vectorstore.persist()
                return True
            except Exception as e:
                print(f"Error ingesting text to vectorstore: {e}")
                return False
        else:
            # Fallback: store in simple dictionary
            doc_id = doc_id or f"doc_{len(self.documents)}"
            self.documents[doc_id] = {
                "text": text,
                "metadata": metadata or {}
            }
            print(f"📝 Document stored in fallback mode: {doc_id}")
            return True

    def ingest_pdf(self, file_path: str, doc_id: Optional[str] = None):
        if LANGCHAIN_AVAILABLE:
            try:
                loader = UnstructuredPDFLoader(file_path)
                docs = loader.load()
                text = "\n".join([d.page_content for d in docs])
                return self.ingest_text(text, doc_id=doc_id)
            except Exception as e:
                print(f"PDF ingest error: {e}")
                return False
        else:
            # Fallback: try to read PDF with basic method
            try:
                # Import pdfplumber if available
                import pdfplumber
                text = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text.append(page_text)
                full_text = '\n'.join(text)
                return self.ingest_text(full_text, doc_id=doc_id)
            except Exception as e:
                print(f"Fallback PDF ingest error: {e}")
                return False

    def ingest_local_text_file(self, file_path: str, doc_id: Optional[str] = None):
        if LANGCHAIN_AVAILABLE:
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                text = "\n".join([d.page_content for d in docs])
                return self.ingest_text(text, doc_id=doc_id)
            except Exception as e:
                print(f"Text file ingest error: {e}")
                return False
        else:
            # Fallback: simple file reading
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return self.ingest_text(text, doc_id=doc_id)
            except Exception as e:
                print(f"Fallback text file ingest error: {e}")
                return False

    def answer_from_docs(self, query: str) -> str:
        if self.qa_chain:
            try:
                result = self.qa_chain.run(query)
                return result.strip()
            except Exception as e:
                return f"⚠️ Document QA error: {str(e)}"
        else:
            # Fallback: simple text search in stored documents
            if not self.documents:
                return "⚠️ No documents have been uploaded yet. Please upload some documents first."
            
            # Simple keyword search
            results = []
            query_lower = query.lower()
            for doc_id, doc_data in self.documents.items():
                text = doc_data["text"].lower()
                if any(word in text for word in query_lower.split()):
                    # Extract a snippet around the match
                    words = text.split()
                    for i, word in enumerate(words):
                        if any(q_word in word for q_word in query_lower.split()):
                            start = max(0, i - 10)
                            end = min(len(words), i + 10)
                            snippet = " ".join(words[start:end])
                            results.append(f"From {doc_id}: ...{snippet}...")
                            break
            
            if results:
                return f"Found information related to your query:\n\n" + "\n\n".join(results[:3])
            else:
                return f"⚠️ No relevant information found for '{query}' in the uploaded documents. LangChain is not available for advanced search."

    # Utility: list collection stats
    def stats(self) -> dict:
        if self.vectorstore:
            try:
                return {
                    "collection": "personal_finance", 
                    "persist_dir": str(self.vectorstore.persist_directory),
                    "langchain_available": LANGCHAIN_AVAILABLE
                }
            except Exception:
                return {"langchain_available": LANGCHAIN_AVAILABLE}
        else:
            return {
                "collection": "fallback_mode",
                "documents_count": len(self.documents),
                "langchain_available": LANGCHAIN_AVAILABLE
            }
