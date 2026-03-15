"""
File: pdf_to_json.py
Author: Hackathon Team
Description:
This utility script converts PDF documents into structured JSON files.
It extracts text from tariff-related PDFs (such as the Mercosur Common Nomenclature
and interpretation rules) and saves them in a cleaned format for later use
in the AI Tariff Agent. The output JSON files are stored in the 'cleaned/' folder
and serve as input for building the FAISS index and classification pipeline.
"""

import os
from PyPDF2 import PdfReader
import json

# Function: Extracts raw text from a PDF file page by page.

def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() + "\n"
    return texto

# Function: Wraps extracted text into a simple JSON structure.
def parse_to_json(texto, nombre_doc):
    estructura = {
        "document": nombre_doc,
        "content": texto
    }
    return estructura

# Function: Saves the JSON structure to disk with UTF-8 encoding.
def guardar_json(estructura, output_path):
    """
    Guarda la estructura JSON en disco con indentación y UTF-8.
    Saves the JSON structure to disk with indentation and UTF-8.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(estructura, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # English: List of PDF files to process (located in the data/ folder)
    pdf_files = [
        ("data/tariff_nomenclature.pdf", "tariff_nomenclature"),
        ("data/interpretation_rules.pdf", "interpretation_rules")
    ]
    # English: Create output folder if it does not exist
    os.makedirs("cleaned", exist_ok=True)

    for pdf_file, nombre_doc in pdf_files:
        # English: Extract text from the PDF
        texto = pdf_to_text(pdf_file)
        
        # English: Convert to JSON
        estructura = parse_to_json(texto, nombre_doc)

        # English: Save JSON file in cleaned/ folder
        output_file = f"cleaned/{nombre_doc}.json"
        guardar_json(estructura, output_file)

        print(f"Procesado {pdf_file} → {output_file}")
