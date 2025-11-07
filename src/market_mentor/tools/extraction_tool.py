""" Tool to extract and analyze text from PDF documents for your MarketBuddy crew."""
import os
import re
from typing import ClassVar
import pypdf
from pydantic import Field
from crewai.tools import BaseTool

class ExtractionTool(BaseTool):
    """
    Tool to extract and analyze text from PDF documents for your MarketBuddy crew.
    """
    # Class constants for tool metadata
    TOOL_NAME: ClassVar[str] = "Extraction_Tool"
    TOOL_DESCRIPTION: ClassVar[str] = "A tool to extract and clean text from PDF documents in the knowledge folder."

    knowledge_folder: str = Field(default="", description="Knowledge folder")
    extracted_output_folder: str = Field(default="", description="Extracted output folder")
   
    def __init__(self):
        super().__init__(
            name=self.TOOL_NAME,
            description=self.TOOL_DESCRIPTION
        )
        
        project_root = os.path.abspath(".")
        self.knowledge_folder = os.path.join(project_root, "knowledge")
        self.extracted_output_folder = os.path.join(project_root, "extracted_output")
        os.makedirs(self.extracted_output_folder, exist_ok=True)
       
    def using_pypdf_extract_text(self, pdf_path: str) -> str:
        """ 
        Extract text from a PDF file using PyPDF. 
        Args:
            pdf_path: Full path to the PDF file
        """
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        return text

    def clean_text(self, extracted_text: str) -> str:
        """ 
        Clean the extracted text by removing unwanted characters and formatting. 
        Args:
            extracted_text: Raw text extracted from PDF
        """
        # Remove hyphenated line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', extracted_text)
        # Merge broken lines  
        text = re.sub(r'\n(?=\w)', ' ', text)  
        # Collapse multiple whitespace   
        text = re.sub(r'\s+', ' ', text)

        return text

    def save_clean_text(self, pdf_filename: str) -> str:
        """ Extract and save clean text from a PDF file.        
        Args:
            pdf_filename: Name of the PDF file (not full path) to process
        """
        pdf_path = os.path.join(self.knowledge_folder, pdf_filename)
        
        # Extract text from PDF
        extracted_text = self.using_pypdf_extract_text(pdf_path)
        cleantext = self.clean_text(str(extracted_text))

        # Create output file path
        extracted_output_filename = pdf_filename.replace('.pdf', '.txt')
        extracted_output_path = os.path.join(self.extracted_output_folder, extracted_output_filename)

        # Save cleaned text
        with open(extracted_output_path, 'w', encoding='utf-8') as f:
            f.write(cleantext)

        return extracted_output_path

    def _run(self, **kwargs) -> str:
        """
        Extract text from PDF documents.
        
        Args:
            **kwargs: Additional parameters ()
        """
        try:
            # Ensure knowledge folder exists
            if not os.path.exists(self.knowledge_folder):
                return f"Error: Knowledge folder not found at {self.knowledge_folder}"

            # Scan knowledge folder for PDF files
            pdf_files = [f for f in os.listdir(self.knowledge_folder) if f.endswith('.pdf')]

            if not pdf_files:
                return f"Error: No PDF files found in {self.knowledge_folder}"

            print(f"Available PDF files in knowledge folder: {pdf_files}")
            
            # Check which files need processing (skip already processed)
            extracted_output_files = []
            skipped_files = []

            for pdf_file in pdf_files:
                extracted_output_filename = pdf_file.replace('.pdf', '.txt')
                extracted_output_path = os.path.join(self.extracted_output_folder, extracted_output_filename)

                # Check if output file already exists
                if os.path.exists(extracted_output_path):
                    print(f"Skipping {pdf_file} - already processed")
                    skipped_files.append(pdf_file)
                    extracted_output_files.append(extracted_output_path)
                else:
                    print(f"Processing PDF file: {pdf_file}")
                    txt_file = self.save_clean_text(pdf_file)
                    extracted_output_files.append(txt_file)
            
            # Build a concise summary that matches the task expected_output format
           
            processed_pairs = []
            for out_path in extracted_output_files:
                # out_path may be absolute or relative; extract filename mapping
                out_fname = os.path.basename(out_path)
                # Attempt to derive source pdf name from text filename
                src_pdf = out_fname.replace('.txt', '.pdf')
                processed_pairs.append(f"{src_pdf} -> extracted_output/{out_fname}")

            summary = f"Processed {len(pdf_files)} PDFs: " + ", ".join(processed_pairs)
            # If any files were skipped, append that info
            if skipped_files:
                summary += f"; Skipped {len(skipped_files)} already-processed files."

            return summary
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"