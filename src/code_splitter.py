import re
from langchain_core.documents import Document

MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ---------------------------
# Generic Chunking Function
# ---------------------------
def chunk_text(text, max_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


# ---------------------------
# C# Method Extractor (Brace-safe)
# ---------------------------
def extract_csharp_methods(code):
    pattern = r'(public|private|protected|internal)\s+(async\s+)?[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{'
    methods = []

    for match in re.finditer(pattern, code):
        start = match.start()
        brace_count = 0
        i = match.end()

        for j in range(i, len(code)):
            if code[j] == '{':
                brace_count += 1
            elif code[j] == '}':
                if brace_count == 0:
                    methods.append(code[start:j+1])
                    break
                brace_count -= 1

    return methods


# ---------------------------
# Extract Class Name
# ---------------------------
def extract_class_name(code):
    match = re.search(r'class\s+(\w+)', code)
    return match.group(1) if match else "UnknownClass"


# ---------------------------
# Extract Method Name
# ---------------------------
def extract_method_name(method_code):
    match = re.search(r'\s+(\w+)\s*\(', method_code)
    return match.group(1) if match else "UnknownMethod"


# ---------------------------
# C# Split Logic
# ---------------------------
def split_csharp_code(code, file_path):
    chunks = []

    class_name = extract_class_name(code)
    methods = extract_csharp_methods(code)

    print(f"[DEBUG] {file_path} → Methods found: {len(methods)}")

    # Method-level chunks (BEST for RAG)
    for method in methods:
        method_name = extract_method_name(method)

        for chunk in chunk_text(method):
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "language": "csharp",
                    "class": class_name,
                    "method": method_name,
                    "type": "method"
                }
            ))

    # Class-level fallback (context)
    if not chunks:
        for chunk in chunk_text(code):
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "language": "csharp",
                    "class": class_name,
                    "type": "class"
                }
            ))

    return chunks


# ---------------------------
# Python Split Logic
# ---------------------------
def split_python_code(code, file_path):
    chunks = []

    class_pattern = r'class\s+(\w+).*?:'
    method_pattern = r'def\s+(\w+)\(.*?\):'

    class_matches = list(re.finditer(class_pattern, code))

    for cls in class_matches:
        class_name = cls.group(1)
        start = cls.start()

        # Find class block (till next class or EOF)
        end = len(code)
        next_classes = [c.start() for c in class_matches if c.start() > start]
        if next_classes:
            end = min(next_classes)

        class_block = code[start:end]

        # Extract methods
        methods = re.finditer(method_pattern, class_block)

        for m in methods:
            method_name = m.group(1)
            method_start = m.start()

            method_end = len(class_block)
            next_methods = [mm.start() for mm in re.finditer(method_pattern, class_block) if mm.start() > method_start]
            if next_methods:
                method_end = min(next_methods)

            method_code = class_block[method_start:method_end]

            for chunk in chunk_text(method_code):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "language": "python",
                        "class": class_name,
                        "method": method_name,
                        "type": "method"
                    }
                ))

    # fallback
    if not chunks:
        for chunk in chunk_text(code):
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "language": "python",
                    "type": "file"
                }
            ))

    return chunks


# ---------------------------
# Entry Point
# ---------------------------
def split_code_into_chunks(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    if file_path.endswith(".cs"):
        return split_csharp_code(code, file_path)

    elif file_path.endswith(".py"):
        return split_python_code(code, file_path)

    else:
        return [Document(
            page_content=code[:MAX_CHUNK_SIZE],
            metadata={"source": file_path, "type": "file"}
        )]