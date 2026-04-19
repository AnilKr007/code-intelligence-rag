
EMBEDDING_MODEL = "nomic-embed-text"  # Keep as is (lightweight)

# Use a much smaller model that fits in 4.7 GB RAM
LLM_MODEL = "phi3:mini"  # 3.8GB - perfect for your memory


CHROMA_DB_DIR = "./db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3