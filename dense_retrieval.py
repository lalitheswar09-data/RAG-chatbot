import faiss
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.index = None
        self._build_index()

    def _build_index(self):
        texts = [chunk["text"] for chunk in self.chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        # Exact cosine similarity search:
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def embed_query(self, query):
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
        ).astype("float32")

        faiss.normalize_L2(embedding)
        return embedding

    def search(self, query, top_k=5):
        query_embedding = self.embed_query(query)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            result = self.chunks[index].copy()
            result["score"] = float(score)
            results.append(result)

        return results


def build_hnsw_index(embeddings, m=32):
    """
    HNSW alternative for larger collections.

    Replace IndexFlatIP with IndexHNSWFlat after creating
    normalized embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(
        dimension,
        m,
        faiss.METRIC_INNER_PRODUCT,
    )
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index
