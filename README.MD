# Customs AI Agent — Intelligent HS Code Classifier - AI Dev Days Hackathon 2026

![Customs AI Agent banner](images/banner_comex.png)

**Slogan:** An explainable HS/NCM classification agent that provides a technical ruling with regulatory justification.

---

## Project Overview

**The Customs AI Agent** is a demonstration AI system that classifies goods according to the Mercosur Common Nomenclature (NCM) and issues a technical ruling citing the General Rules of Interpretation (GRI) and the relevant section, chapter, and subheading notes. The solution combines local FAISS vector indices, Azure OpenAI embeddings and chat models, and a Streamlit interface to provide an interactive and explainable classification workflow for use cases related to customs and trade compliance.

**Who is it for?** Customs officials, trade compliance teams, logistics professionals, and developers building compliance automation solutions.

**What it does**

- Retrieves relevant regulatory text and tariff headings using FAISS.
- Creates a query in English and consults the Azure OpenAI chat to generate a technical ruling.
- Returns: the applicable NCM heading, relevant rules and notes, a reasoned explanation, and a final classification conclusion.
- Provides a link to the official Mercosur nomenclature for manual verification.

---

## Features

- **Explainable classification** citing GRI notes and section/chapter/subheading notes.
- **Local FAISS retrieval** for deterministic context retrieval during the demonstration.
- **Azure OpenAI** for embeddings and LLM-based reasoning.
- **Streamlit user interface** for an interactive demo experience.
- **Reindexing utility** to rebuild FAISS indexes from cleaned JSON sources.

---

## Key technologies used

- **Azure OpenAI**: embeddings and chat inference.
- **Azure AI Foundry**: project provisioned in the subscription (documented in the architecture).
- **Azure Storage account**: available in the environment (indexes stored locally for the demo).
- **FAISS**: local vector search engine.
- **Streamlit**: demo user interface.
- **GitHub**: public repository and development workflow.

> **Note:** FAISS indexes are included locally in `indexes/` to ensure reproducibility during the hackathon demo. In production, indexes can be stored in Azure Storage or a managed vector store and integrated with Foundry.
> 

---

## Architecture (overview)

1. **The user** interacts with the **Streamlit user interface** (`app.py`).
2. The user interface calls the **classifier wrapper** (`src/classifier.py`), which:
    - retrieves relevant documents from **the FAISS index**,
    - creates a request in English,
    - calls **the Azure OpenAI chat** to obtain a technical opinion.
3. **The cleaned data** (`cleaned/`) and **FAISS indexes** (`indexes/`) are included in the repository for the demo.
4. **Azure resources** used or provisioned: Azure OpenAI endpoints, Azure AI Foundry project, storage account (available), Key Vault (recommended for secrets).

**Note on index storage:** For the hackathon demo, FAISS indexes are generated locally and included in `indexes/.` For production, store the indexes in Azure Storage or a managed vector store and connect them to Foundry for scalable retrieval.


![Diagram of the project architecture](docs\diagram.jpg)

---

The demo has been implemented with Streamlit for quick interactive testing. It can be deployed to Azure using standard approaches:

- **Azure App Service (Web App for Containers)**: containerize the Streamlit application and deploy it.
- **Azure Container Instances / AKS**: for scalable container deployments.
- **Key Vault**: recommended for storing Azure OpenAI keys and other secrets.
- **Storage account**: for persistent artifacts and production index storage.

---

## Quickstart guide: setup and run (local demo)

**Prerequisites**

- Python 3.10+
- Azure OpenAI endpoints and keys (for embeddings and chat)
- Optional: Docker for container deployment

**Environment**

1. In `.env`, fill in the required keys:
    - `AZURE_OPENAI_ENDPOINT_TEXT`
    - `AZURE_OPENAI_KEY_TEXT`
    - `AZURE_OPENAI_ENDPOINT_CHAT`
    - `AZURE_OPENAI_KEY_CHAT`
2. Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

- If you want to regenerate the index locally:

```bash
python src/classification_index.py

```

This script reads cleaned JSON files from `the \`cleaned/` directory`, splits the content into granular documents, generates embeddings using Azure OpenAI, and saves a FAISS index in` the `indexes/` directory.`

**Run the Streamlit demo**

```bash
streamlit run app.py
```

### Repository Structure

```bash
.
├─ app.py                      # Streamlit front-end (entry point)
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ src/
│  ├─ pdf_to_json.py           # Convert PDFs to cleaned JSON
│  ├─ classification_index.py  # Build FAISS index from cleaned JSON
│  ├─ classifier.py            # Load index, build prompt, call Azure OpenAI						
├─ data/                       # Original PDF sources
├─ cleaned/                    # Cleaned JSON files used to build the index
├─ indexes/                    # Local FAISS index files (for demo)
├─ docs/                       # architecture diagram, slides, screenshots
└─ images/
    └─ banner_comex.png
    └─ logo.png               # Banner image used by the app and README
```

---

## 📽️ Demo Presentation

You can watch the full project presentation and live demo in the following video:
[Customs AI Agent - Project Presentation](https://youtu.be/4ZCWmMUnNRA)

---

## 🏁 Conclusion

The **Customs AI Agent** demonstrates how generative AI can transform high-precision technical fields like international trade. By combining the reasoning capabilities of **Azure OpenAI** with the deterministic retrieval of **FAISS**, we've created a tool that doesn't just provide a code, but a legally-backed technical ruling. This solution reduces operational risks, minimizes human error, and empowers customs professionals with transparent and explainable AI.

---

## 👤 Author & Recognition

**Fernando Pedernera***Data Analyst | International Trade Expert*

- **LinkedIn:** [fernando-pedernera](https://www.linkedin.com/in/fernando-pedernera/)
- **Microsoft Learn:** [fernando-pedernera (Official Transcript)](https://learn.microsoft.com/es-es/users/fernando-pedernera/transcript/73kn2h22mjk4x2w)the user]