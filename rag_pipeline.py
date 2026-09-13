from dense_retrieval import DenseRetriever
from bm25_retrieval import BM25Retriever
from hybrid_fusion import reciprocal_rank_fusion
from cross_encoder_reranking import CrossEncoderReranker
from generation import generate_answer


class RAGPipeline:
    def __init__(self, chunks):
        self.chunks = chunks

        self.dense = DenseRetriever(chunks)
        self.bm25 = BM25Retriever(chunks)
        self.reranker = CrossEncoderReranker()

    def answer(self, query, strategy="hybrid_reranked"):

        if strategy == "bm25":
            retrieved_chunks = self.bm25.search(
                query,
                top_k=3
            )

        elif strategy == "dense":
            retrieved_chunks = self.dense.search(
                query,
                top_k=3
            )

        elif strategy == "hybrid_rrf":
            dense_results = self.dense.search(
                query,
                top_k=10
            )

            bm25_results = self.bm25.search(
                query,
                top_k=10
            )

            retrieved_chunks = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                k=60,
                top_k=3
            )

        elif strategy == "hybrid_reranked":
            dense_results = self.dense.search(
                query,
                top_k=10
            )

            bm25_results = self.bm25.search(
                query,
                top_k=10
            )

            candidates = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                k=60,
                top_k=10
            )

            retrieved_chunks = self.reranker.rerank(
                query,
                candidates,
                top_k=3
            )

        else:
            raise ValueError(
                f"Unknown retrieval strategy: {strategy}"
            )

        generation_result = generate_answer(
            query,
            retrieved_chunks
        )

        generation_result["retrieved_chunks"] = retrieved_chunks

        return generation_result