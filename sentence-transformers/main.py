corpus_embeddings = corpus_embeddings.to("cuda")
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

query_embeddings = query_embeddings.to("cuda")
query_embeddings = util.normalize_embeddings(query_embeddings)

hits = util.semantic_search(
    query_embeddings,
    corpus_embeddings,
    score_function=util.dot_score,
)