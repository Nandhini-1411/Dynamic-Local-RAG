# Simple RAG Application 📚

A simple **Retrieval-Augmented Generation (RAG)** application built using Python and LangChain.

The application can read **PDF, DOCX, and TXT files**, retrieve relevant information from them, and generate answers using a locally running LLM.

## 🔄 RAG Pipeline

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS
   ↓
Retriever
   ↓
Prompt
   ↓
Ollama
   ↓
Answer
```

## 🛠️ Technologies Used

* Python
* LangChain
* Sentence Transformers (`all-MiniLM-L6-v2`)
* FAISS 
* Ollama (`Llama 3.2:1b`)

## ✨ Features

* Supports PDF, DOCX and TXT files
* Dynamic document loading
* Recursive text chunking
* Semantic search using embeddings
* FAISS vector similarity search
* Local LLM generation using Ollama
* No paid API required

## 📂 Project Structure

```text
Simple_RAG/
│
├── data/
├── create_vector_store.py
├── chat.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add documents

Place your PDF, DOCX, or TXT files inside the `data/` folder.

### 3. Create the vector store

```bash
python create_vector_store.py
```

### 4. Start the chatbot

```bash
python chat.py
```

Make sure Ollama is installed and the required model is available:

```bash
ollama pull llama3.2:1b
```

## 🎯 Purpose

This project was built to understand and implement a complete **RAG pipeline from document ingestion to LLM generation** using locally running models.
