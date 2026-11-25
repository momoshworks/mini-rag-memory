import os
import json
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Chunk:
    def __init__(self, chunk_id: str, source: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.chunk_id = chunk_id
        self.source = source
        self.text = text
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            source=data["source"],
            text=data["text"],
            metadata=data.get("metadata", {})
        )

class SearchResult:
    def __init__(self, chunk: Chunk, score: float):
        self.chunk = chunk
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "text": self.chunk.text,
            "score": round(self.score, 4)
        }

class MiniVectorStore:
    """
    RAG Vector Store with multi-format file extraction:
    - PDF (.pdf) via pypdf
    - Word (.docx) via python-docx
    - Markdown & Plain Text (.md, .txt)
    - Structured data & code (.json, .csv, .py)
    """
    SUPPORTED_EXTENSIONS = (".txt", ".md", ".json", ".csv", ".pdf", ".docx", ".py")

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def extract_text_from_file(self, file_path: str) -> str:
        """Extracts text content based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()

        # 1. PDF
        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                pages_text = []
                for idx, page in enumerate(reader.pages):
                    t = page.extract_text()
                    if t:
                        pages_text.append(f"[صفحة {idx + 1}]\n{t.strip()}")
                return "\n\n".join(pages_text)
            except Exception as e:
                print(f"Error reading PDF {file_path}: {e}")
                return ""

        # 2. Word (.docx)
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                # Extract tables too
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            paras.append(row_text)
                return "\n\n".join(paras)
            except Exception as e:
                print(f"Error reading DOCX {file_path}: {e}")
                return ""

        # 3. Text, Markdown, JSON, CSV
        else:
            for enc in ["utf-8", "utf-8-sig", "windows-1256", "latin-1"]:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def _split_into_chunks(self, text: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', cleaned_text) if p.strip()]
        chunks: List[Chunk] = []
        chunk_idx = 0

        for para in paragraphs:
            if len(para) <= self.chunk_size:
                chunks.append(Chunk(
                    chunk_id=f"{source}_chunk_{chunk_idx}",
                    source=source,
                    text=para,
                    metadata=metadata
                ))
                chunk_idx += 1
            else:
                start = 0
                while start < len(para):
                    end = start + self.chunk_size
                    if end < len(para):
                        space_idx = para.rfind(' ', start, end)
                        if space_idx > start + (self.chunk_size // 2):
                            end = space_idx
                    chunk_text = para[start:end].strip()
                    if chunk_text:
                        chunks.append(Chunk(
                            chunk_id=f"{source}_chunk_{chunk_idx}",
                            source=source,
                            text=chunk_text,
                            metadata=metadata
                        ))
                        chunk_idx += 1
                    start += self.chunk_size - self.chunk_overlap

        return chunks

    def add_text(self, text: str, source: str = "raw_text", metadata: Optional[Dict[str, Any]] = None):
        new_chunks = self._split_into_chunks(text, source, metadata)
        self.chunks.extend(new_chunks)
        self._rebuild_index()

    def add_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        source_name = os.path.basename(file_path)
        content = self.extract_text_from_file(file_path)
        if not content:
            return 0

        new_chunks = self._split_into_chunks(content, source_name, metadata)
        # Remove existing chunks for this source if re-indexing to avoid duplicates
        self.chunks = [c for c in self.chunks if c.source != source_name]
        self.chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self):
        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        corpus = [c.text for c in self.chunks]
        token_pattern = r'(?u)\b[\w\u0600-\u06FF\u0750-\u077F]+\b'
        self.vectorizer = TfidfVectorizer(
            token_pattern=token_pattern,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[SearchResult]:
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1]
        results: List[SearchResult] = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score < min_score:
                break
            results.append(SearchResult(chunk=self.chunks[idx], score=score))
            if len(results) >= top_k:
                break

        return results

    def save(self, file_path: str):
        data = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunks": [c.to_dict() for c in self.chunks]
        }
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunk_size = data.get("chunk_size", 400)
        self.chunk_overlap = data.get("chunk_overlap", 50)
        self.chunks = [Chunk.from_dict(item) for item in data.get("chunks", [])]
        self._rebuild_index()
