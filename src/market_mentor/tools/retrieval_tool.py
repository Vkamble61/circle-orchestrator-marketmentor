""" Retrieval Tool for searching vector database."""
import os
from typing import ClassVar 
from pydantic import Field

# Fallback for version compatibility issues
from crewai.tools import BaseTool
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:    
    from langchain_community.embeddings import OllamaEmbeddings

# Try importing Chroma from langchain_chroma first - version compatibility
try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback for missing langchain_chroma
    from langchain_community.vectorstores import Chroma

class RetrievalTool(BaseTool):
    """
    Tool to retrieve relevant information from the vector database.
    """
    TOOL_NAME: ClassVar[str] = "Retrieval_Tool"
    TOOL_DESCRIPTION: ClassVar[str] = "Search the vector database for relevant documents. Call with any query parameter."   
    
    storage_folder: str = Field(default='', description="Storage folder for vector database")

    def __init__(self):
        super().__init__(
            name=self.TOOL_NAME,
            description=self.TOOL_DESCRIPTION
        )
        # Lazy initialization to avoid loading on import
        self._embeddings = None
        self._vector_store = None
        self._retriever = None
        project_root = os.path.abspath(".")       
        self.storage_folder = os.path.join(project_root, "storage")
    
    def _get_retriever(self):
        """Initialize retriever only when needed."""
        if self._retriever is None:
            try:
                # project_root = os.path.abspath(".")
                # memory_path = os.path.join(project_root, "memory")
                
                print(f"Loading vector store from: {self.storage_folder}")
                
                self._embeddings = OllamaEmbeddings(
                    model="nomic-embed-text", 
                    base_url="http://localhost:11434"
                )
                self._vector_store = Chroma(
                    persist_directory=self.storage_folder,
                    embedding_function=self._embeddings
                )
                self._retriever = self._vector_store.as_retriever(k=3)
                print("Vector store loaded successfully")
                
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return None
                
        return self._retriever

    def _run(self, question: str = "", **kwargs) -> str:
        """Retrieve relevant documents based on query."""
        try:
            # Use default if empty
            if not question or not question.strip():
                question = "Describe the key points of this document."
                print("No valid question found, using default")

            # Clean the question and assign to final variable
            question = question.strip()
            print(f"Final question assigned: '{question}'")

            retriever = self._get_retriever()
            if retriever is None:
                return "Vector database not available. Please ensure indexing is complete."

            print(f"Searching for: {question}")

            # Retrieve documents
            docs = retriever.invoke(question)

            if not docs:
                return "No relevant documents found."
            
            # Format results concisely
            result = f"Retrieved {len(docs)} relevant documents:\n\n"
            
            for i, doc in enumerate(docs, 1):
                # Limit content length for performance
                content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                result += f"Document {i}: {content}\n\n"
            
            return result
            
        except Exception as e:
            return f"Retrieval error: {str(e)}"



