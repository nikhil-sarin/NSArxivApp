"""Vector database for paper embeddings and semantic search."""

import os
import hashlib
import re
from pathlib import Path
import chromadb
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class _HashingEmbedder:
    """Offline fallback that keeps retrieval available without model downloads."""

    dimension = 384

    def encode(self, text: str):
        vector = np.zeros(self.dimension, dtype=float)
        for token in re.findall(r"[a-z0-9-]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


class PaperVectorDB:
    """Vector database for storing and searching paper embeddings."""

    def __init__(self, db_path: str = "data/vector_db"):
        """
        Initialize the vector database.

        Args:
            db_path: Path to store the ChromaDB database.
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.legacy_db_error = ""
        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
        except BaseException as exc:
            # pyo3 PanicException inherits BaseException. Never mutate or delete
            # the legacy index; switch to a clean versioned store and rebuild.
            self.legacy_db_error = str(exc)
            self.db_path = self.db_path.parent / f"{self.db_path.name}_v2"
            print(f"[vector_db] legacy store is incompatible ({exc}); using {self.db_path}")
            self.db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Create collection for papers
        self.collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"},
        )
        # Keep new chunk embeddings out of legacy Chroma databases. Chroma 1.x
        # can read old collections but may panic while altering their schema.
        self.document_db_path = self.db_path.parent / f"{self.db_path.name}_documents"
        self.document_client = chromadb.PersistentClient(path=str(self.document_db_path))
        self.document_collection = self.document_client.get_or_create_collection(
            name="research_documents",
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize sentence transformer for embeddings
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        try:
            self.embedder = SentenceTransformer(embedding_model, device=embedding_device)
            self.embedding_backend = embedding_model
        except Exception as exc:
            print(f"[vector_db] embedding model unavailable ({exc}); using offline hashing fallback")
            self.embedder = _HashingEmbedder()
            self.embedding_backend = "hashing-fallback-v1"

    def add_paper(
        self,
        paper_id: str,
        title: str,
        summary: str,
        metadata: Dict,
        vector: Optional[List[float]] = None,
    ):
        """
        Add a paper to the vector database.

        Args:
            paper_id: Unique identifier for the paper.
            title: Paper title.
            summary: Paper summary.
            metadata: Additional metadata (authors, date, etc.).
            vector: Pre-computed embedding (optional).
        """
        # Generate embedding if not provided
        if vector is None:
            text = f"{title} {summary}"
            vector = self.embedder.encode(text).tolist()

        # Add to collection
        self.collection.add(
            ids=[paper_id],
            embeddings=[vector],
            documents=[summary],
            metadatas=[metadata],
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for papers similar to the query.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of matching papers with metadata.
        """
        # Generate embedding for query
        query_vector = self.embedder.encode(query).tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        papers = []
        if results["ids"] and results["ids"][0]:
            for i, paper_id in enumerate(results["ids"][0]):
                papers.append(
                    {
                        "id": paper_id,
                        "summary": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

        return papers

    def upsert_document(self, document_id: str, text: str, metadata: Dict):
        """Embed and store one attributable research document."""
        safe_metadata = {
            key: value if isinstance(value, (str, int, float, bool)) else str(value)
            for key, value in metadata.items()
            if value is not None
        }
        vector = self.embedder.encode(text).tolist()
        self.document_collection.upsert(
            ids=[document_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[safe_metadata],
        )

    def search_documents(
        self,
        query: str,
        top_k: int = 10,
        paper_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Semantic search across section/page/note/report document records."""
        query_vector = self.embedder.encode(query).tolist()
        where = None
        if paper_ids:
            where = {"paper_id": paper_ids[0]} if len(paper_ids) == 1 else {"paper_id": {"$in": paper_ids}}
        kwargs = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self.document_collection.query(**kwargs)
        documents = []
        if results["ids"] and results["ids"][0]:
            for index, document_id in enumerate(results["ids"][0]):
                documents.append({
                    "document_id": document_id,
                    "text": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "distance": results["distances"][0][index],
                })
        return documents

    def delete_documents(self, owner_type: str, owner_id: str):
        try:
            self.document_collection.delete(where={"owner_key": f"{owner_type}:{owner_id}"})
        except Exception:
            pass

    def get_embedding(self, paper_id: str) -> Optional[List[float]]:
        """Return the stored embedding for a paper, or None if not found."""
        try:
            result = self.collection.get(ids=[paper_id], include=["embeddings"])
            if result["embeddings"] and result["embeddings"][0] is not None:
                return result["embeddings"][0]
        except Exception:
            pass
        return None

    def search_by_vector(self, vector: List[float], top_k: int = 10, exclude_id: Optional[str] = None) -> List[Dict]:
        """Search for papers similar to a given embedding vector."""
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k + (1 if exclude_id else 0),
            include=["documents", "metadatas", "distances"],
        )
        papers = []
        if results["ids"] and results["ids"][0]:
            for i, pid in enumerate(results["ids"][0]):
                if pid == exclude_id:
                    continue
                papers.append({
                    "id": pid,
                    "summary": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
        return papers[:top_k]

    def search_by_metadata(
        self, category: str, top_k: int = 10
    ) -> List[Dict]:
        """
        Search for papers by category.

        Args:
            category: Category to filter by.
            top_k: Number of results to return.

        Returns:
            List of matching papers.
        """
        results = self.collection.get(
            where={"category": category},
            limit=top_k,
            include=["documents", "metadatas"],
        )

        papers = []
        if results["ids"]:
            for i, paper_id in enumerate(results["ids"]):
                papers.append(
                    {
                        "id": paper_id,
                        "summary": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i],
                    }
                )

        return papers

    def get_all_papers(self) -> List[Dict]:
        """Get all papers in the database."""
        results = self.collection.get(include=["documents", "metadatas"])

        papers = []
        if results["ids"]:
            for i, paper_id in enumerate(results["ids"]):
                papers.append(
                    {
                        "id": paper_id,
                        "summary": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i],
                    }
                )

        return papers

    def delete_paper(self, paper_id: str):
        """Remove a paper from the vector database."""
        try:
            self.collection.delete(ids=[paper_id])
        except Exception:
            pass

    def delete_paper(self, paper_id: str):
        """Remove a paper from the vector database."""
        try:
            self.collection.delete(ids=[paper_id])
        except Exception:
            pass

    def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper exists in the database."""
        try:
            result = self.collection.get(ids=[paper_id])
            return bool(result.get("ids"))
        except Exception:
            return False
