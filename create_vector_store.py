from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ================================
# Load Documents
# ================================

DATA_FOLDER = Path("data")
documents = []

for file in DATA_FOLDER.iterdir():

    if file.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file))

    elif file.suffix.lower() == ".txt":
        loader = TextLoader(str(file), encoding="utf-8")

    elif file.suffix.lower() == ".docx":
        loader = Docx2txtLoader(str(file))

    else:
        print(f"Skipping {file.name}")
        continue

    documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

# ================================
# Chunk Documents
# ================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# ================================
# Embedding Model
# ================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================================
# Create FAISS Vector Store
# ================================

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# ================================
# Save Vector Store
# ================================

vector_store.save_local("vector_store")

print("✅ Vector Store Created & Saved Successfully!")
