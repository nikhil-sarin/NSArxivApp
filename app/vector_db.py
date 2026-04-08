"""Vector database for paper embeddings and semantic search."""

from pathlib import Path
import chromadb
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class PaperVectorDB:
    """Vector database for storing and searching paper embeddings."""

    def __init__(self, db_path: str = "vector_db"):
        """
        Initialize the vector database.

        Args:
            db_path: Path to store the ChromaDB database.
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Create collection for papers
        self.collection = self.client.get_or_create_collection(
            name="papers",
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize sentence transformer for embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

    def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper exists in the database."""
        try:
            self.collection.get(ids=[paper_id])
            return True
        except:
            return False
