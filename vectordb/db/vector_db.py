from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Union, overload
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


class Item:
    def __init__(self, *, id=None, score=None, vector=None, payload=None):
        self.id = id
        self.score = score
        self.vector = vector
        self.payload = payload

    def __repr__(self) -> str:
        return f"{{id: {self.id}, score: {self.score}, vector: {self.vector}, payload: {self.payload}}}"


SearchResult = List[Item]


class VectorDB(ABC):
    def __init__(self):
        """Initialize VectorDB"""

    def try_create_collection(self, collection_name: str, dimension: int, measure):
        """Create a collection if not exists"""
        ...

    def add(
        self,
        collection_name,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]] = None,
    ):
        """Add vectors to a collection."""
        ...

    @overload
    def get(self, collection_name:str, filter: dict, top_k=5):
        """Retrieve vectors from a collection."""
        ...

    @overload
    def get(self, collection_name: str, id: str): 
        """Retrieve vectors from a collection."""
        ...

    def search(
        self,
        collection_name: str,
        query: Union[np.ndarray, list[np.ndarray]],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for the top K similar vectors to the query vector."""
        ...

    def delete(self, collection_name: str, id):
        """Delete a vector from collection"""
        ...

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        ...
