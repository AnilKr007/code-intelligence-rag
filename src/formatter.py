# formatter.py

def format_docs_with_metadata(docs):
    formatted = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")

        formatted.append(
            f"[Source: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def print_retrieved_docs(docs):
    print("\nRetrieved Chunks:\n")

    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(doc.page_content)
        print()