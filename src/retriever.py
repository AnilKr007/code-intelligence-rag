# retriever.py

def get_retriever(vectorstore, top_k):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


# Multi-query expansion (basic version)
def expand_query(query):
    variations = [
        query,
        f"Explain: {query}",
        f"Give details about: {query}",
    ]
    return variations