
import os
import pdfplumber
import ast
import pypdf
from pydantic import BaseModel
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Any
#from crewai import tools
from crewai.tools import BaseTool

class ExtractionTool(BaseTool):
    """
    Tool to extract and analyze text from PDF documents for your MarketMentor crew.
    """
    name: str = "PDF_Extraction_Tool"
    description: str = "Tool to extract and analyze text from PDF documents for MarketMentor."
    project_path: ClassVar[Path] = Path.cwd()
    pdf_path: ClassVar[Path] = project_path / 'knowledge' / 'Role_of_GenAI_in_Marketing.pdf'

    
    
    def extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text using pdfplumber - RECOMMENDED METHOD
        Better handling of complex layouts, tables, and formatting
        """
        print("🔄 Extracting with pdfplumber...")

        extracted_data = {
            'method': 'pdfplumber',
            'pages': {},
            'full_text': '',
            'tables': [],
            'metadata': {}
        }

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Get document metadata
                extracted_data['metadata'] = {
                    'total_pages': len(pdf.pages),
                    'pdf_metadata': pdf.metadata
                }

                full_text_parts = []

                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from page
                    page_text = page.extract_text() or ""

                    # Extract tables
                    page_tables = page.extract_tables()

                    # Store page data
                    extracted_data['pages'][f'page_{page_num}'] = {
                        'text': page_text,
                        'tables_count': len(page_tables),
                        'character_count': len(page_text)
                    }

                    # Add to full text with page markers
                    full_text_parts.append(f"=== PAGE {page_num} ===")
                    full_text_parts.append(page_text)

                    # Process tables
                    for table_idx, table in enumerate(page_tables):
                        table_text = f"\\n--- TABLE {table_idx + 1} ON PAGE {page_num} ---\\n"
                        for row in table:
                            if row:  # Skip empty rows
                                row_text = " | ".join([str(cell or "") for cell in row])
                                table_text += row_text + "\\n"

                        extracted_data['tables'].append({
                            'page': page_num,
                            'table_number': table_idx + 1,
                            'content': table_text
                        })
                        full_text_parts.append(table_text)

                extracted_data['full_text'] = "\\n".join(full_text_parts)

            print(f"✅ pdfplumber: Extracted {extracted_data['metadata']['total_pages']} pages")
            print(f"📊 Found {len(extracted_data['tables'])} tables")

        except Exception as e:
            print(f"❌ pdfplumber extraction failed: {e}")

        return extracted_data
        # Any additional initialization if needed
    
    def format_for_agents(self, extracted_data: Dict[str, Any]) -> str:
        """
        Format extracted data for use by agents
        """
        formatted_parts = []
        
        # Add header
        formatted_parts.append(f"DOCUMENT ANALYSIS")
        formatted_parts.append(f"Extraction Method: {extracted_data.get('method', 'unknown')}")
        formatted_parts.append(f"Total Pages: {extracted_data.get('metadata', {}).get('total_pages', 'unknown')}")
        formatted_parts.append("=" * 50)
        
        # Add page content
        pages = extracted_data.get('pages', {})
        for page_key, page_data in pages.items():
            page_num = page_key.replace('page_', '')
            formatted_parts.append(f"\\nPAGE {page_num}:")
            formatted_parts.append("-" * 20)
            formatted_parts.append(page_data.get('text', ''))
        
        # Add tables if any
        tables = extracted_data.get('tables', [])
        if tables:
            formatted_parts.append("\\nTABLES FOUND:")
            formatted_parts.append("-" * 20)
            for table in tables:
                formatted_parts.append(table.get('content', ''))
        
        return "\\n".join(formatted_parts)
    
    def extract_company_document(self, pdf_path: str) -> str:
        """        Main function to extract text         """
        
        if not os.path.exists(pdf_path):
            return f"❌ File not found: {pdf_path}"
        
        try:
            # Get the best extraction
            extracted_data = self.extract_with_pdfplumber(pdf_path)
            
            # Format for agents
            formatted_content = self.format_for_agents(extracted_data)
            
            print(f"📋 Extraction Summary:")
            print(f"   Method: {extracted_data.get('method')}")
            print(f"   Pages: {extracted_data.get('metadata', {}).get('total_pages')}")
            print(f"   Characters: {len(formatted_content)}")
            print(f"   Tables: {len(extracted_data.get('tables', []))}")
            
            return formatted_content
            
        except Exception as e:
            return f"❌ Extraction failed: {str(e)}"

    
    def _run(self) -> str:
        try:
            
            # Use pdfplumber for extraction
            extracted_data = self.extract_company_document(str(self.pdf_path))
            return str(extracted_data)
                    
        
        except Exception as e:
            return f"❌ Error during extraction: {str(e)}"

    
    

