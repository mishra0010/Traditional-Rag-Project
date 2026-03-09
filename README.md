# Traditional RAG Project – PDF Question Answering System

## Overview

This project implements a **Traditional Retrieval-Augmented Generation (RAG) pipeline** that allows users to ask questions from PDF documents. Instead of relying solely on the language model’s training data, the system retrieves relevant information from the uploaded documents and provides it as context to the LLM for generating accurate answers.

The project demonstrates how **vector databases, embeddings, and LLMs** can be combined to build a **document-based question answering system**.

---

# Project Architecture

```
PDF Document
     ↓
Document Loader (PyPDFLoader)
     ↓
Document Objects
     ↓
Text Chunking (RecursiveCharacterTextSplitter)
     ↓
Text Chunks
     ↓
Tokenization (Internal to Embedding Model)
     ↓
Embedding Generation (Sentence Transformers)
     ↓
Vector Embeddings
     ↓
Vector Storage (FAISS)
     ↓
Retriever
     ↓
User Query
     ↓
Query Embedding
     ↓
Similarity Search in FAISS
     ↓
Top-K Relevant Chunks Retrieved
     ↓
Context + Query Sent to LLM (ChatGroq)
     ↓
Final Answer Generated
```

---

# Tech Stack

| Component            | Technology Used                               |
| -------------------- | --------------------------------------------- |
| Programming Language | Python                                        |
| Framework            | LangChain                                     |
| Environment          | Jupyter Notebook                              |
| Document Loader      | PyPDFLoader                                   |
| Text Splitter        | RecursiveCharacterTextSplitter                |
| Embedding Model      | Sentence Transformers (HuggingFaceEmbeddings) |
| Vector Database      | FAISS                                         |
| LLM                  | ChatGroq                                      |
| Data Source          | PDF Documents                                 |

---

# Key Components Explained

## 1. Document Loading

The first step is to load the PDF file and extract text from it.

**Tool Used:** PyPDFLoader

Purpose:

* Reads PDF documents
* Extracts text page by page
* Converts them into structured `Document` objects

Example code:

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()
```

Example output:

```
Document(
 page_content="Machine learning is a subset of AI...",
 metadata={"page":1}
)
```

---

# 2. Text Chunking

Large documents cannot be directly processed by embedding models or LLMs. Therefore, the text is split into smaller chunks.

**Tool Used:** RecursiveCharacterTextSplitter

Example:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
```

### Why Chunking is Important

* Embedding models have input size limits
* Smaller chunks improve retrieval accuracy
* Overlap preserves context across chunks

Example:

```
Chunk 1 → characters 0–1000
Chunk 2 → characters 800–1800
Chunk 3 → characters 1600–2600
```

---

# 3. Tokenization

Before embeddings are generated, **tokenization takes place internally inside the embedding model**.

Example:

```
Text:
"Machine learning is powerful"

Tokenization:
["machine", "learning", "is", "powerful"]

Token IDs:
[4521, 7812, 24, 9981]
```

These tokens are then processed by the transformer model to generate embeddings.

---

# 4. Embedding Generation

Each chunk of text is converted into a **vector embedding**.

**Tool Used:** Sentence Transformers via HuggingFaceEmbeddings

Example code:

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Example transformation:

```
Text Chunk:
"Machine learning enables computers to learn from data"

Vector Embedding:
[0.213, -0.443, 0.781, 0.112, ...]
```

These vectors represent the **semantic meaning of the text**.

---

# 5. Vector Database

The generated embeddings are stored in a vector database for similarity search.

**Tool Used:** FAISS (Facebook AI Similarity Search)

Example code:

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
```

FAISS builds an index that allows efficient **nearest neighbor search** between vectors.

---

# 6. Retriever

The FAISS vector store is converted into a retriever interface.

Example:

```python
retriever = vectorstore.as_retriever()
```

The retriever performs:

```
User Query
     ↓
Convert Query to Embedding
     ↓
Search Similar Vectors in FAISS
     ↓
Return Top-K Relevant Chunks
```

---

# 7. Query Processing

When the user asks a question:

1. The query is converted into an embedding using the same embedding model.
2. FAISS compares the query vector with stored document vectors.
3. The most similar chunks are retrieved.

Example:

```
Query:
"What is machine learning?"

Retrieved Chunks:
1. Definition of machine learning
2. Explanation of supervised learning
3. Overview of neural networks
```

---

# 8. Context Augmentation

The retrieved chunks are combined with the user query to form a **context-aware prompt**.

Example prompt:

```
Context:
Machine learning is a subset of artificial intelligence...

Question:
What is machine learning?
```

This ensures that the LLM answers **based on the document content**.

---

# 9. Answer Generation

The final step is generating the answer using an LLM.

**LLM Used:** ChatGroq

The LLM receives:

```
User Question
+
Retrieved Context
```

It then generates a response grounded in the retrieved document chunks.

Example output:

```
Machine learning is a subset of artificial intelligence that allows systems to learn patterns from data and improve performance without explicit programming.
```

---

# Complete Workflow

```
User Uploads PDF
        ↓
Text Extracted Using PyPDFLoader
        ↓
Text Split into Chunks
        ↓
Embeddings Generated Using Sentence Transformers
        ↓
Embeddings Stored in FAISS
        ↓
User Asks Question
        ↓
Query Converted to Embedding
        ↓
FAISS Performs Similarity Search
        ↓
Top Relevant Chunks Retrieved
        ↓
Context + Question Sent to ChatGroq
        ↓
LLM Generates Final Answer
```

---

# Why RAG is Useful

Retrieval-Augmented Generation improves LLM responses by:

* Reducing hallucinations
* Using external knowledge sources
* Providing document-grounded answers
* Enabling domain-specific question answering

---

# Key Concepts Demonstrated

This project demonstrates:

* Document ingestion pipelines
* Text chunking strategies
* Transformer-based embeddings
* Vector similarity search
* Retrieval-Augmented Generation (RAG)
* Integration of vector databases with LLMs

---

# Example Use Cases

* PDF Question Answering
* Knowledge Base Chatbots
* Research Paper Querying
* Enterprise Document Search
* AI Assistants for Documentation

---

# Future Improvements

Possible enhancements to the project:

* Add hybrid search (keyword + vector search)
* Implement reranking models for better retrieval
* Add a web interface using Streamlit
* Use production-grade vector databases (Pinecone, Weaviate, Milvus)
* Implement conversational memory

---

# Conclusion

This project demonstrates a complete **Traditional RAG pipeline using LangChain, Sentence Transformers, FAISS, and ChatGroq**. The system efficiently retrieves relevant information from documents and uses it to generate accurate answers using a large language model.

It showcases how **LLMs can be augmented with external knowledge sources**, making them more reliable and useful for real-world applications.
