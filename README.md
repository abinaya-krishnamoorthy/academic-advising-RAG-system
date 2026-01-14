# academic-advising-RAG-system
A Retrieval-Augmented Generation AI system built with LangChain and Chroma DB for academic advising queries. Includes data ingestion pipeline, chunking strategies, embeddings, and Streamlit UI.
# Academic Advising RAG System
A Retrieval-Augmented Generation (RAG) application designed to answer academic advising questions using university documents such as degree plans, course descriptions, program requirements, and curriculum worksheets.

This system was built using **Python, LangChain, ChromaDB, and Streamlit**, and demonstrates full development of a real-world RAG pipeline including document ingestion, chunking, vectorization, retrieval, and LLM-based answering.

---

## 🚀 Project Overview
The goal of this project is to create an AI assistant that helps students quickly find accurate advising information without manually searching through multiple documents. The system retrieves relevant context from a unified vector database and generates grounded answers.

The project includes:

- A unified data ingestion + preprocessing pipeline  
- Chunking strategies for PDF, Word, Excel, and text files  
- Embedding and vector storage using **ChromaDB**  
- Retrieval pipelines (Similarity + MMR)  
- A Streamlit user interface  
- Timeout handling, no-answer safety, and retrieval-only mode  

---

## 🧠 Key Features

### ✔ Multi-format Document Ingestion
Handles:
- PDF files  
- Word (.docx)  
- Excel sheets (.xlsx)  
- Text files  

Data is cleaned, chunked, and stored into a single vector database.

---

### ✔ Custom Chunking Strategies
Implemented efficient chunking for:
- Tabular Excel data  
- Structured PDFs  
- Free-text Word documents  

Each document type uses an optimized approach to maintain context quality.

---

### ✔ Embeddings & Vector Store
- Embedding model: **nomic-embed-text**  
- Vector store: **ChromaDB**  
- Persistent directory with reproducible state  

---

### ✔ Retrieval Pipelines
Supports:
- **Similarity search**  
- **MMR (Max Marginal Relevance)** for diverse results  

With adjustable parameters (via Streamlit UI):
- `top_k`  
- `fetch_k`  
- `lambda` (diversity vs relevance)  

---

### ✔ Streamlit Application
User interface includes:
- Query input box  
- Model selection  
- Retrieval method selector  
- Adjustable retrieval parameters  
- Retrieval-only mode  
- Display of retrieved chunks and context  

---

### ✔ LLM Response Module
- Uses ChatOllama models (e.g., `llama3.2:3b`)  
- Includes system prompts for grounded, context-based answers  
- Timeout guardrails to prevent long-running queries  
- Falls back to “information unavailable in the context” when needed  

---

## 📁 Repository Structure

```text
academic-advising-RAG-system/
│
├── app.py                     # Streamlit UI
├── build_index_single.py      # Full data ingestion & indexing pipeline
├── requirements.txt           # Python dependencies
├── chroma/                    # Vector store directory (gitignored in practice)
├── data/
│   ├── raw/                   # Original documents
│   └── processed/             # Cleaned + chunked data
├── screenshots/               # UI images and sample outputs
└── README.md                  # Project documentation
