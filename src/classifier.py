"""
File: classifier.py
Author: Hackathon Team
Description:
This module defines the classification logic for the AI Tariff Agent.
It loads the FAISS index, initializes Azure OpenAI embeddings and chat models,
and provides functions to classify merchandise according to the Mercosur Common Nomenclature (NCM).
The classification applies the General Rules of Interpretation (GRI) and relevant section/chapter notes,
returning a technical opinion with justification and references.
"""


import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Cargar índice FAISS de clasificación
INDEX_DIR = "indexes/faiss_classification_index"
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    api_version="2024-10-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_TEXT"),
    api_key=os.getenv("AZURE_OPENAI_KEY_TEXT"),
    chunk_size=50
)
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", k=4)

# Inicializar modelo de chat
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    api_version="2024-10-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_CHAT"),
    api_key=os.getenv("AZURE_OPENAI_KEY_CHAT"),
    temperature=0
)

def classify_merchandise(query: str):
    # Recuperar fragmentos relevantes
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    # Construir prompt para el LLM
    # Build prompt for the LLM
    prompt = f"""
    You are an expert in international trade and customs tariff classification.
    Your task is to issue a technical opinion on the merchandise: {query}.
    You must apply the General Rules of Interpretation (GRI), as well as the relevant Section, Chapter, and Subheading notes.

    Consider that:
    - Section, Chapter, and Subheading notes take absolute priority over the main function of the product.
    - Always justify the classification by indicating the applicable NCM heading, the relevant rules, and exclusions from other headings.
    - Explain your reasoning as a customs specialist would, citing Section, Chapter, and notes when necessary.
    - If there are doubts between multiple headings, apply the GRIs in order (1 to 6) and justify the final choice.

    Use the context retrieved from the Mercosur Common Nomenclature (NCM).

    Return:
    - The applicable NCM heading
    - The relevant rules (GRI and Section/Chapter/Subheading notes)
    - A clear and well-founded explanation
    - A final conclusion with the correct classification

    Context:
    {context}
    """


    response = llm.invoke(prompt)
    return response.content

def answer(query: str):
    """
    Wrapper for the Streamlit app.
    Returns the technical opinion and the official Mercosur nomenclature URL for manual verification.
    """
    result = classify_merchandise(query)
    sources = ["https://www.mercosur.int/politica-comercial/nomenclatura-comun-ncm-y-arancel-externo-comun-aec"]
    return result, sources


if __name__ == "__main__":
    query = input("Which merchandise do you want to classify? ")
    result = classify_merchandise(query)
    print("\n--- Classification Result ---\n")
    print(result)
    print("\nOfficial source: https://www.mercosur.int/politica-comercial/nomenclatura-comun-ncm-y-arancel-externo-comun-aec")



