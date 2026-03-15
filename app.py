"""
File: app.py
Author: Hackathon Team
Description:
This is the Streamlit front-end for the AI Tariff Agent.
It provides a user interface to classify merchandise according to the Mercosur Common Nomenclature (NCM).
Users can enter product descriptions, configure Azure environment settings, and receive a technical opinion
with a link to the official Mercosur nomenclature for manual verification.
"""


import streamlit as st
from src.classifier import answer  # imports the wrapper that returns classification + link

# --- Header with image and title ---
st.image("images/banner_comex.png", width=800)  # use your existing image
st.title("Customs Classification Agent – Hackathon Demo")
st.markdown("""
This agent applies the **General Rules of Interpretation (GRI)** and the Section, Chapter, and Subheading notes 
to issue a technical opinion on the tariff classification of goods.
""")

# --- Sidebar with options ---
st.sidebar.header("⚙️ Options")
st.sidebar.markdown("Configure your environment and data")
azure_endpoint = st.sidebar.text_input("Azure Endpoint", value="https://<your-endpoint>.openai.azure.com/")
api_key = st.sidebar.text_input("Azure API Key", type="password")
deployment = st.sidebar.text_input("Deployment Name", value="text-embedding-3-small")

if st.sidebar.button("Reindex data"):
    st.sidebar.success("Reindex completed (simulated).")

# --- Main input ---
query = st.text_input("Which merchandise do you want to classify?")
if st.button("Classify"):
    if query.strip():
        # Call the classifier
        result, sources = answer(query)

        # --- Results panel ---
        st.subheader("📊 Technical Opinion")
        st.write(result)

        st.subheader("🔗 Official Source")
        for src in sources:
            st.markdown(f"[Consult the Mercosur nomenclature]({src})")
    else:
        st.warning("Please enter a description of the merchandise.")
