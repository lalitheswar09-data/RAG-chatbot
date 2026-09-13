from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=3):
        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["rerank_score"] = float(score)
            reranked.append(result)

        return sorted(
            reranked,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )[:top_k]
