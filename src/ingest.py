# ingest_code.py

import os
from langchain_community.vectorstores import Chroma

from utils import get_embedding_model
from config import CHROMA_DB_DIR
from code_splitter import split_code_into_chunks


#Supported file types (extend anytime)
SUPPORTED_EXTENSIONS = (".cs", ".py")


def load_code_from_repo(repo_path="repo"):
    all_chunks = []

    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.join(root, file)

                try:
                    print(file_path)
                    chunks = split_code_into_chunks(file_path)
                    all_chunks.extend(chunks)

                    print(f"Processed: {file_path} ({len(chunks)} chunks)")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return all_chunks


def main():
    print("Scanning full repository...")

    docs = load_code_from_repo(repo_path=".\\sample\\test\\test")

    print(f"\nTotal chunks created: {len(docs)}")

    embedding = get_embedding_model()

    print("Creating embeddings...")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=CHROMA_DB_DIR
    )

    vectorstore.persist()

    print("Full repository ingestion complete!")


if __name__ == "__main__":
    main()