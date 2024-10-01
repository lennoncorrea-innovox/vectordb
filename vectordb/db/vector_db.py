from abc import ABC
from enum import Enum
from typing import List, Union
import numpy as np


class Distance(str, Enum):
    """
    Type of internal tags, build from payload Distance function types used to compare vectors
    """

    def __str__(self) -> str:
        return str(self.value)

    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"
    MANHATTAN = "Manhattan"
    HAMMING = "Hamming"
    # TODO: Check if more distances are needed


class SearchItem:
    def __init__(self, id, score):
        self.id = id
        self.score = score

    def __repr__(self) -> str:
        return f"{{id: {self.id}, score: {self.score}}}"

    id: str
    score: float


SearchResult = List[SearchItem]


class VectorDB(ABC):
    def __init__(self):
        """Initialize VectorDB"""

    def create_collection(self, collection_name: str, dimension: int, measure):
        """Create a collection"""
        ...

    def add(self, collection_name, vectors: np.ndarray):
        """Add vectors to a collection."""
        ...

    def get(self, collection_name: str, id: Union[str, List[str]]) -> List[np.ndarray]:
        """Retrieve vectors from a collection."""
        ...

    def search(
        self,
        collection_name: str,
        query_vector: Union[np.ndarray, list[np.ndarray]],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for the top K similar vectors to the query vector."""
        ...

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        ...
