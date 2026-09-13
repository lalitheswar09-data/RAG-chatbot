import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

        tokenized_documents = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, top_k=5):
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index in indices[:top_k]:
            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results
