def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=60,
    top_k=10,
):
    fused = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **result,
                "rrf_score": 0.0,
            }

        fused[chunk_id]["rrf_score"] += 1 / (k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **result,
                "rrf_score": 0.0,
            }

        fused[chunk_id]["rrf_score"] += 1 / (k + rank)

    return sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )[:top_k]


def min_max_normalize(scores):
    if not scores:
        return {}

    minimum = min(scores.values())
    maximum = max(scores.values())

    if minimum == maximum:
        return {key: 1.0 for key in scores}

    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in scores.items()
    }


def weighted_score_fusion(
    dense_results,
    bm25_results,
    beta=0.5,
    top_k=10,
):
    """
    beta = 1.0 -> dense only
    beta = 0.0 -> BM25 only
    beta = 0.5 -> equal weight
    """
    dense_scores = {
        x["chunk_id"]: x["score"]
        for x in dense_results
    }

    bm25_scores = {
        x["chunk_id"]: x["score"]
        for x in bm25_results
    }

    dense_norm = min_max_normalize(dense_scores)
    bm25_norm = min_max_normalize(bm25_scores)

    lookup = {
        x["chunk_id"]: x
        for x in dense_results + bm25_results
    }

    fused = []

    for chunk_id in set(dense_scores) | set(bm25_scores):
        score = (
            beta * dense_norm.get(chunk_id, 0.0)
            + (1 - beta) * bm25_norm.get(chunk_id, 0.0)
        )

        result = lookup[chunk_id].copy()
        result["fusion_score"] = score
        fused.append(result)

    return sorted(
        fused,
        key=lambda x: x["fusion_score"],
        reverse=True,
    )[:top_k]
