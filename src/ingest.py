# ingest.py
import os
from langchain_chroma import Chroma  # Changed from langchain_community

from utils import get_embedding_model
from config import CHROMA_DB_DIR
from code_splitter import split_code_semantic

SUPPORTED_EXTENSIONS = (".cs", ".py")


def load_code_from_repo(repo_path="repo"):
    all_chunks = []

    for root, dirs, files in os.walk(repo_path):
        # Skip build folders
        if '\\obj\\' in root or '\\bin\\' in root:
            continue

        for file in files:
            if file.endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.join(root, file)

                try:
                    print(f"\nProcessing: {file_path}")
                    chunks = split_code_semantic(file_path)

                    if chunks:
                        all_chunks.extend(chunks)
                        print(f"Created {len(chunks)} chunks")
                    else:
                        print(f"No chunks created")

                except Exception as e:
                    print(f"Error: {str(e)}")

    return all_chunks


def main():
    print("Scanning full repository...")

    docs = load_code_from_repo(repo_path=".\\sample\\test\\test")

    print(f"\n{'=' * 50}")
    print(f"Total chunks created: {len(docs)}")
    print(f"{'=' * 50}\n")

    if not docs:
        print("No documents to index! Check your repository path.")
        return

    embedding = get_embedding_model()

    print("Creating embeddings...")

    # Create vectorstore - it automatically persists to CHROMA_DB_DIR
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=CHROMA_DB_DIR
    )

    # No need for vectorstore.persist() - it's automatic
    print("✓ Full repository ingestion complete!")
    print(f"✓ Vector database saved to: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()