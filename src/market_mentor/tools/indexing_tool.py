"""Indexing tool for processing text documents and creating vector database."""
import os
from typing import ClassVar

from crewai.tools import BaseTool
from pydantic import Field

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Try importing OllamaEmbeddings from langchain_ollama first - version compatibility
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

class IndexingTool(BaseTool):
    """
    Tool for indexing documents and creating a vector database.
    """
    
    TOOL_NAME: ClassVar[str] = "Indexing_Tool"
    TOOL_DESCRIPTION: ClassVar[str] = "A tool for indexing documents"

    extracted_output_folder: str = Field(default='', description="Ingested Files folder")
    storage_folder: str = Field(default='', description="Storage folder for vector database")

    def __init__(self):
        super().__init__(
            name=self.TOOL_NAME,
            description=self.TOOL_DESCRIPTION
        )
        project_root = os.path.abspath(".")
        self.extracted_output_folder = os.path.join(project_root, "extracted_output")
        self.storage_folder = os.path.join(project_root, "storage")


    def _run(self, **kwargs) -> str:
        """
        Run the Indexing tool to index text documents efficiently.
        
        Args:
            path: Optional path to specific text file or folder
            **kwargs: Additional parameters
        """
        try:
            # Process all txt files in the output folder (where extraction tool saves files)
            if not os.path.exists(self.extracted_output_folder):            
                return f"Error: Extracted output folder not found at {self.extracted_output_folder}. Please run PDF extraction first."

            txt_files = [f for f in os.listdir(self.extracted_output_folder) if f.endswith('.txt')]

            if not txt_files:
                return f"Error: No text files found in {self.extracted_output_folder}. Please run PDF extraction first."

            print(f"Indexing {len(txt_files)} text files from {self.extracted_output_folder}")

            # Initialize embeddings and text splitter once (not per file!)
            embeddings = OllamaEmbeddings(
                model='nomic-embed-text',
                base_url="http://localhost:11434"
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            
            # Collect all documents first
            all_documents = []
            processed_files = []
            
            for i, txt_file in enumerate(txt_files, 1):
                print(f"Processing file {i}/{len(txt_files)}: {txt_file}")

                file_path = os.path.join(self.extracted_output_folder, txt_file)

                # Load and split document
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                docs = text_splitter.split_documents(documents)
                
                # Add source metadata
                for doc in docs:
                    doc.metadata['source_file'] = txt_file
                
                all_documents.extend(docs)
                processed_files.append(txt_file)
            
            print(f"Creating vector database with {len(all_documents)} document chunks...")
            
            # Create vector database once with all documents
            
            os.makedirs(self.storage_folder, exist_ok=True)
            
            db = Chroma.from_documents(
                all_documents,
                embeddings,
                persist_directory=self.storage_folder
            )
            db.persist()
            
            print("Vector database created and persisted successfully!")
            
            return f"Successfully indexed {len(processed_files)} files with {len(all_documents)} chunks. Files: {processed_files}"
        except Exception as e:
            return f"Error processing text files: {str(e)}"