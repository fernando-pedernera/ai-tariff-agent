"""
File: classification_index.py
Author: Hackathon Team
Description:
This module builds the FAISS index used for customs classification in the AI Tariff Agent.
It loads cleaned JSON files containing tariff nomenclature and interpretation rules,
splits them into granular documents (tariff codes and rules), and generates embeddings
using Azure OpenAI. The resulting FAISS index enables efficient similarity search
to support classification queries based on the Mercosur Common Nomenclature (NCM).
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

CLEANED_DIR = Path("cleaned")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)

# Function: Loads cleaned JSON files and splits content into granular documents (tariff codes and rules).

def load_json_files():
    docs = []
    for file in CLEANED_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            content = data.get("content", "")

            # Divide by NCM codes (example: 0106.12.00)
            partidas = re.split(r"(?=\d{4}\.\d{2}\.\d{2})", content)
            for p in partidas:
                if p.strip():
                    docs.append(Document(
                        page_content=p.strip(),
                        metadata={"source": file.name, "type": "partida"}
                    ))

            # Divide by rules
            reglas = re.split(r"(?=Regla aplicable)", content)
            for r in reglas:
                if r.strip():
                    docs.append(Document(
                        page_content=r.strip(),
                        metadata={"source": file.name, "type": "regla"}
                    ))

    print(f"Granular documents loaded: {len(docs)}")
    return docs

# Function: Generates embeddings with Azure OpenAI and builds a FAISS index from documents.
def build_faiss_index(docs):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small",
        api_version="2024-10-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_TEXT"),
        api_key=os.getenv("AZURE_OPENAI_KEY_TEXT"),
        chunk_size=10
    )
    print("Generating embeddings and building FAISS index (classification)...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def save_index(vectorstore, name="faiss_classification_index"):
    save_path = INDEX_DIR / name
    vectorstore.save_local(str(save_path))
    print(f"--- Classification index saved in: {save_path} ---")

if __name__ == "__main__":
    docs = load_json_files()
    if docs:
        vectorstore = build_faiss_index(docs)
        save_index(vectorstore)
        print("\n--- Process completed successfully: Classification index created ---")
    else:
        print("No documents were found to process.")

