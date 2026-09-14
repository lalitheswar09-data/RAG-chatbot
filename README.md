# RAG Chatbot

> A modular, end-to-end Retrieval-Augmented Generation system combining
> dense retrieval, lexical search, hybrid fusion, cross-encoder
> reranking, and grounded LLM generation.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-00A67E)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Grounded_Generation-412991?logo=openai&logoColor=white)](https://openai.com/)

## Live Demo

**[Launch the RAG Chatbot →](https://ragchatbotlaith.streamlit.app/)**

------------------------------------------------------------------------

## Overview

This project implements a complete Retrieval-Augmented Generation
pipeline for question answering over a custom document collection.

Instead of relying entirely on an LLM's parametric knowledge, the system
first retrieves relevant passages from an indexed knowledge base and
then supplies those passages to the language model as grounded context.

The architecture combines two complementary retrieval approaches:

-   **Dense retrieval** using Sentence Transformers + FAISS
-   **Lexical retrieval** using BM25

Their results can then be combined with **Reciprocal Rank Fusion (RRF)**
and refined using a **cross-encoder reranker** before the final answer
is generated.

The application exposes these retrieval strategies through an
interactive Streamlit interface, making it possible to inspect how
different retrieval approaches affect the final context and answer.

------------------------------------------------------------------------

## Architecture

``` text
DOCUMENT COLLECTION
        |
        v
+-----------------------+
|       CHUNKING        |
| Structure-aware       |
| recursive splitting   |
| + overlap + metadata  |
+-----------+-----------+
            |
            v
     CHUNKED DOCUMENTS
            |
      +-----+-----+
      |           |
      v           v
+-----------+ +-----------+
|   Dense   | |   BM25    |
| Retrieval | | Retrieval |
| Sentence  | | Lexical   |
|Transformers| | Search   |
|   FAISS   | |           |
+-----+-----+ +-----+-----+
      |             |
      +------+------+
             |
             v
+-----------------------+
|    HYBRID FUSION      |
| Reciprocal Rank       |
| Fusion (RRF)          |
+-----------+-----------+
            |
            v
+-----------------------+
|  CROSS-ENCODER        |
|      RERANKER         |
| Fine-grained scoring  |
+-----------+-----------+
            |
            v
+-----------------------+
|    GROUNDED LLM       |
| Query + Top Context   |
|       OpenAI API      |
+-----------+-----------+
            |
            v
     GROUNDED ANSWER
     + CHUNK CITATIONS
```

------------------------------------------------------------------------

## Retrieval Strategies

### 1. Dense Retrieval

Semantic retrieval using Sentence Transformer embeddings and FAISS.

``` text
Query
  ↓
Embedding
  ↓
FAISS similarity search
  ↓
Top-k semantic chunks
```

### 2. BM25

Traditional lexical retrieval based on token-level term matching.

``` text
Query
  ↓
Tokenization
  ↓
BM25 scoring
  ↓
Top-k lexical matches
```

BM25 is particularly useful when exact terminology, identifiers, names,
or domain-specific keywords matter.

### 3. Hybrid RRF

Dense and BM25 retrieval are executed independently and their rankings
are combined using **Reciprocal Rank Fusion**.

For a document appearing at rank `r`:

``` text
RRF Score = 1 / (k + r)
```

The final ranking aggregates evidence from both retrieval systems.

### 4. Hybrid + Cross-Encoder Reranking

The final strategy retrieves a larger candidate pool using hybrid RRF
and then applies a cross-encoder to score query-document relevance
directly.

``` text
Dense Top-10
     +
BM25 Top-10
     |
     v
   RRF
     |
     v
Hybrid Top-10
     |
     v
Cross-Encoder
     |
     v
Final Top-3
     |
     v
LLM Generation
```

This separates **candidate retrieval** from **fine-grained relevance
scoring**.

------------------------------------------------------------------------

## Chunking

The project uses a lightweight, dependency-minimal chunking module
rather than relying on a framework-specific text splitter.

The splitting strategy follows a hierarchy:

``` text
Document
   |
   v
Paragraphs
   |
   +-- fits → keep
   |
   +-- too large
          |
          v
       Sentences
          |
          +-- fits → keep
          |
          +-- too large
                 |
                 v
               Words
```

Chunks retain metadata required for traceability:

``` python
{
    "chunk_id": "chunk_0001",
    "text": "...",
    "source_doc": "sample.txt",
    "char_start": 0,
    "char_end": 800
}
```

The chunking module supports configurable chunk size, overlap,
source-document tracking, and character offsets.

------------------------------------------------------------------------

## Grounded Generation

The generation stage receives:

``` text
User Query
     +
Retrieved Context
     ↓
OpenAI Model
     ↓
Grounded Answer
```

The generation prompt enforces:

1.  Use only the supplied retrieved context.
2.  Do not rely on outside knowledge.
3.  State when the retrieved context is insufficient.
4.  Cite factual claims using chunk IDs.
5.  Never invent chunk identifiers.

Example:

``` text
The retrieval pipeline combines semantic and lexical search [chunk_0004].
```

The application also identifies which retrieved chunks were cited by the
generated response.

------------------------------------------------------------------------

## Project Structure

``` text
RAG-chatbot/
|
├── chat_app.py
├── rag_pipeline.py
|
├── chunking.py
├── dense_retrieval.py
├── bm25_retrieval.py
├── hybrid_fusion.py
├── cross_encoder_reranking.py
├── generation.py
|
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── run.ipynb
|
└── data/
    └── documents/
        └── sample.txt
```

### Module Responsibilities

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `chunking.py`                       Recursive document chunking and
                                      metadata generation

  `dense_retrieval.py`                Sentence Transformer embeddings and
                                      FAISS retrieval

  `bm25_retrieval.py`                 Lexical BM25 retrieval

  `hybrid_fusion.py`                  Reciprocal Rank Fusion

  `cross_encoder_reranking.py`        Cross-encoder candidate reranking

  `generation.py`                     Grounded OpenAI generation and
                                      citation extraction

  `rag_pipeline.py`                   End-to-end pipeline orchestration

  `chat_app.py`                       Streamlit application and
                                      interactive UI

  `run.ipynb`                         Development and experimentation
                                      notebook
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Tech Stack

### Retrieval

-   **FAISS** --- vector similarity search
-   **Sentence Transformers** --- dense embeddings
-   **BM25** --- lexical retrieval
-   **Reciprocal Rank Fusion** --- hybrid ranking
-   **Cross-Encoder** --- relevance reranking

### Generation

-   **OpenAI API** --- grounded response generation

### Application

-   **Streamlit** --- interactive web application

### Core

-   **Python**
-   **NumPy**
-   **python-dotenv**

------------------------------------------------------------------------

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/lalitheswar09-data/RAG-chatbot.git
cd RAG-chatbot
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate on Windows:

``` bash
.venv\Scripts\activate
```

Or on macOS/Linux:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure the OpenAI API key

Create a `.env` file in the project root:

``` env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-nano
```

**Never commit `.env` to GitHub.**

The repository includes `.env.example` as a template.

------------------------------------------------------------------------

## Add Your Documents

Place `.txt` documents inside:

``` text
data/documents/
```

For example:

``` text
data/
└── documents/
    ├── document_1.txt
    ├── document_2.txt
    └── document_3.txt
```

The application reads the document collection and constructs the
chunking and retrieval pipeline.

------------------------------------------------------------------------

## Run the Application

``` bash
streamlit run chat_app.py
```

The application will open in your browser.

------------------------------------------------------------------------

## Using the Application

The interface allows you to select:

``` text
Dense
BM25
Hybrid RRF
Hybrid + Cross-Encoder Reranking
```

You can then ask questions about the indexed documents.

The application displays:

-   Generated answer
-   Retrieval strategy
-   Retrieved source chunks
-   Source document
-   Chunk identifiers
-   Grounding citations

This makes the retrieval → reranking → generation process inspectable
rather than treating the RAG system as a black box.

------------------------------------------------------------------------

## Example Pipeline

For a query such as:

``` text
What retrieval methods are used in this system?
```

the application performs:

``` text
1. Receive query
        ↓
2. Retrieve semantic candidates with FAISS
        ↓
3. Retrieve lexical candidates with BM25
        ↓
4. Combine rankings with RRF
        ↓
5. Rerank hybrid candidates with cross-encoder
        ↓
6. Select top-3 chunks
        ↓
7. Construct grounded prompt
        ↓
8. Generate answer with OpenAI
        ↓
9. Extract valid chunk citations
        ↓
10. Display answer + retrieved sources
```

------------------------------------------------------------------------

## Design Goals

### Modularity

Each major RAG component is isolated into its own module so retrieval,
fusion, reranking, and generation can be modified independently.

### Inspectability

Retrieved chunks are exposed in the UI, allowing the evidence behind an
answer to be inspected directly.

### Retrieval Diversity

Dense and lexical retrieval solve different failure modes. Combining
them provides broader retrieval coverage than relying on a single
method.

### Grounded Generation

The LLM is explicitly instructed to answer only from retrieved context
and provide chunk-level citations.

### Framework Independence

The core retrieval and chunking components are implemented directly
rather than hiding the system behind a high-level RAG framework.

------------------------------------------------------------------------

## Why Hybrid Retrieval?

Dense and lexical retrieval have complementary strengths.

  Retrieval       Strength
  --------------- ---------------------------------------------
  Dense / FAISS   Semantic similarity and paraphrased queries
  BM25            Exact terms, keywords, identifiers
  Hybrid RRF      Combines both ranking signals
  Cross-Encoder   Fine-grained query-document relevance

The resulting architecture is:

``` text
Candidate Recall
      ↓
Dense + Lexical Retrieval
      ↓
Ranking Fusion
      ↓
Fine-Grained Reranking
      ↓
Context Selection
      ↓
Grounded Generation
```

------------------------------------------------------------------------

## Security

API credentials should never be stored in source code or committed to
the repository.

Local development uses `.env`, while deployment uses Streamlit's secrets
configuration.

The `.gitignore` file prevents accidental commits of `.env`,
`__pycache__/`, and `*.pyc`.

------------------------------------------------------------------------

## Deployment

The application is deployed using **Streamlit Community Cloud**.

``` text
Repository: lalitheswar09-data/RAG-chatbot
Branch: main
Entry point: chat_app.py
Python: 3.12
```

Secrets are configured separately from the GitHub repository.

### Live Application

**https://ragchatbotlaith.streamlit.app/**

------------------------------------------------------------------------

## Future Improvements

Potential extensions include:

-   Document loaders for PDF, Markdown, and HTML
-   Persistent vector indexes
-   Metadata filtering
-   Query rewriting
-   Multi-query retrieval
-   Parent-document retrieval
-   Semantic chunking
-   HNSW / ANN indexing for larger collections
-   Retrieval evaluation datasets
-   Automated retrieval benchmarking
-   Streaming generation
-   Conversation memory
-   Multi-document source attribution

------------------------------------------------------------------------

## Author

**Lalith Eswar Adatarvu**

Computer Science Engineering --- IIIT Naya Raipur

-   GitHub: [lalitheswar09-data](https://github.com/lalitheswar09-data)
-   LinkedIn
-   LeetCode

------------------------------------------------------------------------

## License

This project is intended for educational, portfolio, and experimentation
purposes.
