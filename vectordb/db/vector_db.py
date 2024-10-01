from abc import ABC
from enum import Enum
from typing import Union
import numpy as np

IdResponse = list[list[str]]
DistanceResponse = list[list[float]]


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


class VectorDB(ABC):
    def __init__(self):
        """Initialize a VectorDB ..."""

    def create_collection(self, collection_name: str, dimension: int, measure):
        """Create a collection"""
        ...

    def add(self, collection_name, vectors: np.ndarray):
        """Add vectors to a collection."""
        ...

    def search(
        self,
        collection_name: str,
        query_vector: Union[np.ndarray, list[np.ndarray]],
        top_k: int = 5,
    ):
        """Search for the top K similar vectors to the query vector."""
        ...

    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        ...
