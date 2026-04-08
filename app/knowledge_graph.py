"""Knowledge graph for tracking paper connections."""

import networkx as nx
from typing import List, Dict, Set, Optional
from collections import defaultdict


class KnowledgeGraph:
    """Knowledge graph for paper connections and relationships."""

    def __init__(self):
        """Initialize the knowledge graph."""
        self.graph = nx.Graph()
        self.paper_metadata: Dict[str, Dict] = {}

    def add_paper(self, paper_id: str, metadata: Dict):
        """Add a paper node to the graph."""
        if not self.graph.has_node(paper_id):
            self.graph.add_node(paper_id, **metadata)
            self.paper_metadata[paper_id] = metadata

    def connect_by_category(self, paper_id: str, categories: List[str]):
        """Connect paper to other papers in the same categories."""
        for category in categories:
            # Find other papers in this category
            for node, data in self.graph.nodes(data=True):
                if node != paper_id and category in data.get("categories", []):
                    # Add edge if not exists
                    if not self.graph.has_edge(paper_id, node):
                        self.graph.add_edge(
                            paper_id,
                            node,
                            relation="same_category",
                            category=category,
                        )

    def connect_by_author(self, paper_id: str, authors: List[str]):
        """Connect paper to other papers by same authors."""
        for author in authors:
            # Find other papers by this author
            for node, data in self.graph.nodes(data=True):
                if node != paper_id:
                    node_authors = data.get("authors", [])
                    if author in node_authors:
                        if not self.graph.has_edge(paper_id, node):
                            self.graph.add_edge(
                                paper_id, node, relation="same_author", author=author
                            )

    def get_connected_papers(
        self, paper_id: str, max_depth: int = 2
    ) -> List[tuple]:
        """
        Get papers connected to the given paper.

        Args:
            paper_id: The paper to find connections for.
            max_depth: Maximum depth of connections.

        Returns:
            List of (connected_paper_id, relation_type, relation_data) tuples.
        """
        connections = []

        for neighbor in self.graph.neighbors(paper_id):
            edge_data = self.graph[paper_id][neighbor]
            connections.append((neighbor, edge_data.get("relation", "unknown"), edge_data))

        # Expand to second degree
        if max_depth >= 2:
            for neighbor in list(self.graph.neighbors(paper_id)):
                for second_neighbor in self.graph.neighbors(neighbor):
                    if second_neighbor != paper_id:
                        edge_data = self.graph[neighbor][second_neighbor]
                        connections.append(
                            (
                                second_neighbor,
                                f"via_{neighbor}_{edge_data.get('relation', 'unknown')}",
                                edge_data,
                            )
                        )

        return connections

    def get_similar_papers(
        self, paper_id: str, vector_db, top_k: int = 5
    ) -> List[Dict]:
        """
        Get similar papers based on vector similarity.

        Args:
            paper_id: The paper to find similar papers for.
            vector_db: Vector database instance.
            top_k: Number of similar papers to return.

        Returns:
            List of similar papers.
        """
        metadata = self.paper_metadata.get(paper_id, {})
        title = metadata.get("title", "")
        summary = metadata.get("summary", "")

        # Search for similar papers
        similar = vector_db.search(f"{title} {summary}", top_k=top_k + 1)

        # Remove the original paper from results
        similar = [p for p in similar if p["id"] != paper_id]

        return similar

    def get_paper_info(self, paper_id: str) -> Optional[Dict]:
        """Get metadata for a paper."""
        return self.paper_metadata.get(paper_id)

    def get_all_paper_ids(self) -> List[str]:
        """Get all paper IDs in the graph."""
        return list(self.graph.nodes())

    def get_papers_by_category(self, category: str) -> List[str]:
        """Get all papers in a specific category."""
        return [
            node
            for node, data in self.graph.nodes(data=True)
            if category in data.get("categories", [])
        ]

    def to_dict(self) -> Dict:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": [
                {
                    "id": node,
                    "data": dict(self.graph.nodes[node]),
                }
                for node in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "data": dict(self.graph.edges[edge]),
                }
                for edge in self.graph.edges()
            ],
        }
