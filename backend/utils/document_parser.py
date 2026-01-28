"""
Document Parser Utilities
Extract text from PDF, DOCX, DOC files
"""

import os
import fitz  # PyMuPDF
import pythoncom
import zipfile
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import fromstring
from docx import Document
from win32com.client import Dispatch
from fastapi import HTTPException
from io import BytesIO


def extract_pdf_text(file_stream):
    """Extract text from PDF using PyMuPDF"""
    try:
        # Use PyMuPDF to open the BytesIO stream
        with fitz.open("pdf", file_stream) as pdf:
            if not pdf.pages:
                raise HTTPException(status_code=400, detail="The PDF has no pages.")

            # Extract text from all pages
            combined_text = []
            for i, page in enumerate(pdf):
                page_text = page.get_text()
                if page_text.strip():
                    combined_text.append(f"--- Page {i + 1} ---\n{page_text.strip()}")

            if combined_text:
                return "\n\n".join(combined_text)
            else:
                raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")


def extract_docx_text(file_stream):
    """Extract text from DOCX using python-docx and XML parsing"""
    # First try python-docx (more robust for many files)
    try:
        doc = Document(BytesIO(file_stream))
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        if paragraphs:
            return "\n".join(paragraphs)
    except Exception:
        # Fall back to XML extraction below
        pass

    try:
        # Read the stream into BytesIO
        docx_bytes = BytesIO(file_stream)

        # Open the DOCX file as a ZIP archive
        with zipfile.ZipFile(docx_bytes, 'r') as docx_zip:
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

            # Initialize list for all extracted text
            all_text = []

            # Function to extract text from an XML part
            def extract_text_from_xml(xml_content):
                root = fromstring(xml_content)
                text_parts = []
                for paragraph in root.findall('.//w:p', namespaces):
                    texts = [node.text for node in paragraph.findall('.//w:t', namespaces) if node.text]
                    if texts:
                        text_parts.append(''.join(texts))
                return text_parts

            # Extract main document text
            if 'word/document.xml' in docx_zip.namelist():
                document_xml = docx_zip.read('word/document.xml').decode('utf-8', errors='ignore')
                main_text = extract_text_from_xml(document_xml)
                all_text.extend(main_text)

            # Extract headers
            header_files = [f for f in docx_zip.namelist() if f.startswith('word/header')]
            headers_text = []
            for header_file in header_files:
                header_xml = docx_zip.read(header_file).decode('utf-8', errors='ignore')
                headers_text.extend(extract_text_from_xml(header_xml))
            if headers_text:
                all_text.append("\nHeaders:")
                all_text.extend(headers_text)

            # Extract text boxes (in the main document)
            if 'word/document.xml' in docx_zip.namelist():
                document_xml = docx_zip.read('word/document.xml').decode('utf-8', errors='ignore')
                root = fromstring(document_xml)
                text_boxes = []
                for text_box in root.findall('.//w:txbxContent', namespaces):
                    texts = [node.text for node in text_box.findall('.//w:t', namespaces) if node.text]
                    if texts:
                        text_boxes.append('Text Box: ' + ''.join(texts))
                if text_boxes:
                    all_text.append("\nText Boxes:")
                    all_text.extend(text_boxes)

            return "\n".join(all_text) if all_text else None

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from DOCX: {e}")


def extract_doc_text(file_stream):
    """Extract text from DOC using COM (Windows only)"""
    try:
        pythoncom.CoInitialize()  # Initialize the COM library for thread support

        # Save the byte stream to a temporary file
        temp_file_path = os.path.join(os.getcwd(), "temp_doc_file.doc")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_stream)
        
        # Dispatch the Word application and open the document
        word_app = Dispatch('Word.Application')
        word_app.Visible = False  # Run Word in the background
        doc = word_app.Documents.Open(temp_file_path)
        doc_text = doc.Content.Text  # Extract the text from the document

        # Close the document and Word application
        doc.Close()
        word_app.Quit()

        # Delete the temporary file
        os.remove(temp_file_path)

        return doc_text.strip()  # Return the cleaned text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from DOC: {e}")
    finally:
        pythoncom.CoUninitialize()  # Uninitialize the COM library


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Extract text from uploaded file based on extension
    
    Args:
        file_content: Raw file bytes
        filename: Original filename with extension
    
    Returns:
        Extracted text content
    """
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext == 'pdf':
        return extract_pdf_text(file_content)
    elif file_ext == 'docx':
        return extract_docx_text(file_content)
    elif file_ext == 'doc':
        return extract_doc_text(file_content)
    elif file_ext == 'txt':
        return file_content.decode('utf-8')
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: PDF, DOCX, DOC, TXT"
        )
