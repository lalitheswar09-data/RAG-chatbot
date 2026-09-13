# RAG Project — Separate Files

Each component is intentionally separated so you can study and test it independently.

## Files

1. `01_chunking.py` — paragraph/sentence/word chunking
2. `02_eval_dataset.py` — generate RAG evaluation questions
3. `03_dense_retrieval.py` — Sentence Transformers + FAISS
4. `04_generation.py` — grounded OpenAI generation
5. `05_ragas_evaluation.py` — RAGAS metrics
6. `06_bm25_retrieval.py` — BM25
7. `07_hybrid_fusion.py` — RRF and weighted fusion
8. `08_cross_encoder_reranking.py` — cross-encoder reranking
9. `09_benchmark.py` — benchmark four pipelines
10. `10_chat_app.py` — Streamlit chat UI
11. `rag_pipeline.py` — connects the components

## Important

The OpenAI model is configured through:

```env
OPENAI_MODEL=gpt-5-nano
```

If that model ID is not available to your API account/current API catalog, replace it with an available GPT-5 model. The OpenAI model catalog should be checked before running.

## Folder layout

```text
rag-project/
├── data/
│   ├── documents/
│   │   └── *.txt
│   └── eval_dataset.json
├── 01_chunking.py
├── 02_eval_dataset.py
├── 03_dense_retrieval.py
├── 04_generation.py
├── 05_ragas_evaluation.py
├── 06_bm25_retrieval.py
├── 07_hybrid_fusion.py
├── 08_cross_encoder_reranking.py
├── 09_benchmark.py
├── 10_chat_app.py
├── rag_pipeline.py
├── requirements.txt
└── .env.example
```

## Recommended order

Run/study them in order:

Chunking → Dense → BM25 → Fusion → Reranking → Generation → RAGAS → Benchmark → Chat.

To launch the chat:

```bash
streamlit run 10_chat_app.py
```
