# 🚀 Personal Knowledge Assistant

A full-stack Retrieval-Augmented Generation (RAG) system that allows you to chat with your local documents (PDFs and Text files). Built with modern AI techniques including Semantic Chunking and Metadata-Rich Chunking to handle complex cross-document queries.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Backend:** FastAPI, Python
* **LLM Core:** LangChain, Groq (Llama-3.1-8b)
* **Embeddings & Vector Store:** HuggingFace (`BAAI/bge-small-en-v1.5`), FAISS

## ✨ Features
* **Semantic Chunking:** Groups document sentences by their underlying mathematical meaning rather than an arbitrary character count limit.
* **Metadata-Rich Chunking:** Injects document source titles directly into the semantic chunks, preventing "Document Confusion" when querying across multiple PDFs.


## 📦 Installation & Setup

1. **Install Requirements**
   ```bash
   pip install fastapi uvicorn pydantic langchain langchain-groq langchain-huggingface langchain-community langchain-experimental faiss-cpu streamlit requests
   ```

2. **Add Your Documents**
   Place your `.pdf` or `.txt` files inside the `knowledge_base` folder.

3. **Set your API Key**
   Ensure your Groq API key is set inside `app.py`. *(Note: Never commit your API key to GitHub! It is highly recommended to use a `.env` file instead).*

## 🚀 Running the App

You need to run two separate terminal windows to start the system.

**Terminal 1 (Start the Backend API):**
```bash
uvicorn app:app --reload
```
*(The backend runs on `http://localhost:8000`. The first time you run it, it will take a moment to parse your PDFs and build the `faiss_index` folder).*

**Terminal 2 (Start the Frontend UI):**
```bash
streamlit run ui.py
```
*(The frontend will automatically open in your browser at `http://localhost:8501`).*

## 🧹 Troubleshooting
If you add new PDFs to the knowledge base or change the chunking logic in the code, you **must delete** the `faiss_index` folder to force the system to rebuild its memory:
```bash
rm -rf faiss_index
```

## 📊 Evaluation & Metrics
This project includes a custom-built, zero-dependency evaluation framework (`evaluate.py`) that tests the RAG pipeline against a set of Ground Truth questions.

### Current Performance (Multi-Document Test)
The system was tested against a cross-document knowledge base (multiple resumes) achieving the following scores:

| Metric | Score | Description |
|--------|-------|-------------|
| **Context Recall** | **1.00 / 1.00** | The retriever successfully fetched the correct source document for 100% of questions. |
| **Faithfulness** | **0.89 / 1.00** | The AI answers are highly grounded in the retrieved documents without hallucination. |
| **Answer Relevancy** | **0.87 / 1.00** | The generated answers directly address the core terms and intent of the user's questions. |

### How to run the evaluation:
1. Ensure the backend is running (`uvicorn app:app`).
2. Run the evaluation script:
   ```bash
   python3 evaluate.py
   ```
