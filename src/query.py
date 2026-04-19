from langchain_chroma import Chroma  # Changed from langchain_community
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import get_embedding_model
from config import CHROMA_DB_DIR, LLM_MODEL, TOP_K
from retriever import get_retriever, expand_query
from formatter import format_docs_with_metadata, print_retrieved_docs


def main():
    print("Loading vector DB...")

    embedding = get_embedding_model()

    # Load existing vectorstore (no persist() method needed)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding
    )

    retriever = get_retriever(vectorstore, TOP_K)

    # Use the smaller model from config
    llm = Ollama(model=LLM_MODEL)

    prompt = ChatPromptTemplate.from_template("""
You are a code assistant. Answer ONLY from the provided context.

If answer is not found, say "I don't know".

Context:
{context}

Question:
{question}

Mention the source file names in your answer.
""")

    chain = prompt | llm | StrOutputParser()

    while True:
        query = input("\nAsk a question (or 'exit'): ")

        if query.lower() == "exit":
            break

        # Multi-query expansion
        queries = expand_query(query)

        all_docs = []
        for q in queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        # Remove duplicates
        seen = set()
        unique_docs = []

        for doc in all_docs:
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        # Debug
        print_retrieved_docs(unique_docs)

        context = format_docs_with_metadata(unique_docs)

        try:
            response = chain.invoke({
                "context": context,
                "question": query
            })

            print("\nFinal Answer:")
            print(response)
        except Exception as e:
            print(f"\nError: {e}")
            print("Try a different question or check if Ollama is running")


if __name__ == "__main__":
    main()