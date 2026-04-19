import re
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ---------------------------
# Simple but Effective Code-Aware Chunker (No SemanticChunker issues)
# ---------------------------
class CodeAwareChunker:
    def __init__(self, max_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.max_size = max_size
        self.overlap = overlap

    def chunk_code(self, code, metadata):
        """Chunk code preserving structure when possible"""
        chunks = []

        # First, split by logical code blocks (methods, classes)
        blocks = self._split_by_code_blocks(code)

        for block in blocks:
            if len(block) <= self.max_size:
                # Small enough, keep as is
                chunks.append(Document(page_content=block, metadata=metadata))
            else:
                # Too large, split by lines with overlap
                sub_chunks = self._chunk_by_lines(block)
                for sub_chunk in sub_chunks:
                    chunks.append(Document(page_content=sub_chunk, metadata=metadata))

        return chunks

    def _split_by_code_blocks(self, code):
        """Split by C# or Python code blocks"""
        # Split by method/function boundaries
        method_pattern = r'(?=\n\s*(?:public|private|protected|internal|static|async|def|class)\s+)'
        blocks = re.split(method_pattern, code)

        # Filter empty blocks
        blocks = [b.strip() for b in blocks if b.strip()]

        # Merge small blocks with previous
        merged = []
        current = ""

        for block in blocks:
            if len(current) + len(block) < self.max_size and current:
                current += "\n\n" + block
            else:
                if current:
                    merged.append(current)
                current = block

        if current:
            merged.append(current)

        return merged if merged else [code]

    def _chunk_by_lines(self, text):
        """Simple line-based chunking with overlap"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            if current_size + len(line) > self.max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                # Keep overlap lines
                overlap_lines = current_chunk[-self.overlap // 10:]
                current_chunk = overlap_lines
                current_size = sum(len(l) for l in current_chunk)

            current_chunk.append(line)
            current_size += len(line)

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks


# ---------------------------
# Improved C# Extractor (Fixed)
# ---------------------------
def extract_csharp_methods_safe(code):
    """Extract C# methods safely without infinite loops"""
    methods = []

    # Find class name
    class_match = re.search(r'class\s+(\w+)', code)
    class_name = class_match.group(1) if class_match else "UnknownClass"

    # Find namespace
    namespace_match = re.search(r'namespace\s+([\w\.]+)', code)
    namespace = namespace_match.group(1) if namespace_match else ""

    # Method pattern
    method_pattern = r'(?:\[.*?\]\s*)?(?:public|private|protected|internal)\s+(?:static\s+)?(?:async\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{'

    for match in re.finditer(method_pattern, code):
        method_name = match.group(1)
        start = match.start()

        # Find matching closing brace
        brace_count = 0
        i = match.end() - 1  # Start at the opening brace

        for j in range(i, len(code)):
            if code[j] == '{':
                brace_count += 1
            elif code[j] == '}':
                brace_count -= 1
                if brace_count == 0:
                    method_code = code[start:j + 1]
                    methods.append({
                        "content": method_code,
                        "class": class_name,
                        "method": method_name,
                        "namespace": namespace
                    })
                    break

    return methods, class_name, namespace


# ---------------------------
# Improved Python Extractor (Fixed)
# ---------------------------
def extract_python_functions_safe(code):
    """Extract Python functions/classes safely"""
    chunks = []

    # Find all classes
    class_pattern = r'class\s+(\w+)(?:\([^)]*\))?:'

    if re.search(class_pattern, code):
        # Has classes
        for class_match in re.finditer(class_pattern, code):
            class_name = class_match.group(1)
            class_start = class_match.start()

            # Find class end (next class at same indent level)
            lines = code[class_start:].split('\n')
            class_lines = []
            base_indent = None

            for line in lines:
                if line.strip() and base_indent is None:
                    base_indent = len(line) - len(line.lstrip())
                elif line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                    break
                class_lines.append(line)

            class_block = '\n'.join(class_lines)

            # Extract methods
            method_pattern = r'\n(\s+)def\s+(\w+)\s*\([^)]*\):'
            for method_match in re.finditer(method_pattern, class_block):
                method_name = method_match.group(2)
                method_start = method_match.start()

                # Find method end
                method_lines = []
                method_indent = len(method_match.group(1))

                rest_lines = class_block[method_start:].split('\n')
                for line in rest_lines:
                    if line.strip() and len(line) - len(line.lstrip()) <= method_indent and method_lines:
                        break
                    method_lines.append(line)

                method_code = '\n'.join(method_lines)

                chunks.append({
                    "content": method_code,
                    "class": class_name,
                    "method": method_name
                })
    else:
        # No classes, extract functions
        func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
        for func_match in re.finditer(func_pattern, code):
            method_name = func_match.group(1)
            func_start = func_match.start()

            # Find function end
            lines = code[func_start:].split('\n')
            func_lines = []
            base_indent = None

            for line in lines:
                if line.strip() and base_indent is None:
                    base_indent = len(line) - len(line.lstrip())
                elif line.strip() and len(line) - len(line.lstrip()) <= base_indent and func_lines:
                    break
                func_lines.append(line)

            func_code = '\n'.join(func_lines)

            chunks.append({
                "content": func_code,
                "class": "",
                "method": method_name
            })

    return chunks


# ---------------------------
# Main splitting function (No SemanticChunker)
# ---------------------------
def split_code_semantic(file_path):
    """Split code using structure-aware chunking"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        if not code.strip():
            print(f"Empty file: {file_path}")
            return []

        chunker = CodeAwareChunker()
        all_docs = []

        if file_path.endswith(".cs"):
            methods, class_name, namespace = extract_csharp_methods_safe(code)

            if methods:
                print(f"  ✓ Found {len(methods)} methods in {file_path}")
                for method in methods:
                    docs = chunker.chunk_code(method["content"], {
                        "source": file_path,
                        "language": "csharp",
                        "class": method["class"],
                        "method": method["method"],
                        "namespace": method["namespace"],
                        "type": "method"
                    })
                    all_docs.extend(docs)
            else:
                # No methods found, chunk entire file
                print(f"No methods found, chunking entire file: {file_path}")
                docs = chunker.chunk_code(code, {
                    "source": file_path,
                    "language": "csharp",
                    "type": "class",
                    "class": class_name
                })
                all_docs.extend(docs)

        elif file_path.endswith(".py"):
            functions = extract_python_functions_safe(code)

            if functions:
                print(f"  ✓ Found {len(functions)} functions in {file_path}")
                for func in functions:
                    docs = chunker.chunk_code(func["content"], {
                        "source": file_path,
                        "language": "python",
                        "class": func["class"],
                        "method": func["method"],
                        "type": "method"
                    })
                    all_docs.extend(docs)
            else:
                # No functions found, chunk entire file
                print(f"No functions found, chunking entire file: {file_path}")
                docs = chunker.chunk_code(code, {
                    "source": file_path,
                    "language": "python",
                    "type": "file"
                })
                all_docs.extend(docs)

        else:
            # Unknown file type
            docs = chunker.chunk_code(code, {
                "source": file_path,
                "language": "unknown",
                "type": "file"
            })
            all_docs.extend(docs)

        return all_docs

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []


# ---------------------------
# Legacy compatibility function
# ---------------------------
def split_code_into_chunks(file_path):
    """Legacy function for compatibility"""
    return split_code_semantic(file_path)