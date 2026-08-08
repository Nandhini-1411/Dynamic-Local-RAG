from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# ================================
# Load Embedding Model
# ================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================================
# Load Saved Vector Store
# ================================

vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

# ================================
# Create Retriever
# ================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ================================
# Load Ollama
# ================================

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# ================================
# Prompt Template
# ================================

prompt = PromptTemplate.from_template(
"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not available in the context, reply:

"I couldn't find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ================================
# Chat Loop
# ================================

while True:

    question = input("\nAsk a Question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    print("\n" + "=" * 80)
    print("Answer:")
    print("=" * 80)
    print(response.content)
