import logging
from typing import List, Dict, Any, Tuple, Union
import numpy as np
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    PointStruct,
    PointIdsList,
    SearchRequest,
    Filter,
    FieldCondition,
    MatchValue,
    Distance as QdrantDistance,
)
from .vector_db import (
    Distance,
    SearchItem,
    SearchResult,
    VectorDB,
)


class Qdrant(VectorDB):
    def __init__(self, host: str = "localhost", port: int = 6333):
        self._client = QdrantClient(host=host, port=port)

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: Distance = Distance.COSINE,
    ):
        qdrant_distance = self._parse_distance(distance)

        if self._client.collection_exists(collection_name):
            logging.warning(f"Index for collection '{collection_name}' already exists.")
            return

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=qdrant_distance),
        )
        logging.info(
            f"Collection '{collection_name}' created with vector size {dimension} and distance {qdrant_distance}."
        )

    def add(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]] = None,
    ):
        if payloads is None:
            payloads = [{} for _ in vectors]

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=payload,
            )
            for i, (vector, payload) in enumerate(zip(vectors, payloads))
        ]
        self._client.upsert(collection_name=collection_name, points=points, wait=False)
        logging.info(f"Added {len(vectors)} vectors to collection '{collection_name}'.")

    def get(self, collection_name: str, id) -> List[np.ndarray]:
        if isinstance(id, list):
            return self._get_many(collection_name, id)
        return self._get(collection_name, id)

    def search(
        self,
        collection_name: str,
        query: Union[np.ndarray, List[np.ndarray]],
        top_k: int = 5,
        filter: dict = None,
    ) -> List[SearchResult]:
        if isinstance(query, list):
            return self._search_vectors(collection_name, query, top_k, filter)

        return self._search_vector(collection_name, query, top_k, filter)

    def update(
        self,
        collection_name: str,
        id: str,
        vector: np.ndarray,
        payload: Dict[str, Any] = None,
    ):
        if isinstance(vector, np.ndarray):
            if vector.ndim == 2:
                vector = vector.squeeze(0)

        point = PointStruct(
            id=id, vector=vector, payload=payload if payload is not None else {}
        )
        self._client.upsert(collection_name=collection_name, points=[point], wait=False)
        logging.info(f"Updated vector with id {id} in collection '{collection_name}'.")

    def delete(self, collection_name: str, id):
        try:
            points_selector = PointIdsList(points=[id])
            self._client.delete(
                collection_name=collection_name,
                points_selector=points_selector,
                wait=False,
            )
            logging.info(
                f"Deleted vector with id {id} from collection '{collection_name}'."
            )
        except Exception as e:
            logging.error(f"Error deleting vector: {e}")

    def delete_collection(self, collection_name: str):
        if self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name=collection_name)
            logging.info(f"Deleted collection '{collection_name}'.")

    def _search_vector(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        filter: dict = None,
    ):
        if isinstance(query_vector, np.ndarray):
            if query_vector.ndim == 2:
                query_vector = query_vector.squeeze(0)

        if filter:
            filter = self._parse_filter(filter)

        search_result = self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_vectors=False,
            query_filter=filter,
        )

        if not search_result:
            return None

        search_result = [
            SearchItem(result.id, result.score) for result in search_result
        ]

        return [search_result]

    def _search_vectors(
        self,
        collection_name: str,
        query_vectors: List[np.ndarray],
        top_k: int = 5,
        filter: dict = None,
    ):
        if filter:
            filter = self._parse_filter(filter)

        search_requests = [
            SearchRequest(
                vector=np.array(vector),
                limit=top_k,
                with_vector=False,
                filter=filter,
            )
            for vector in query_vectors
        ]

        search_results = self._client.search_batch(
            collection_name=collection_name, requests=search_requests
        )

        if not search_results or not search_results[0]:
            return None

        search_results = [
            [SearchItem(result.id, result.score) for result in results]
            for results in search_results
        ]

        return search_results

    def _get(self, collection_name: str, id):
        result = self._client.retrieve(
            collection_name, [id], with_vectors=True, with_payload=False
        )
        if result:
            result = np.array(result[0].vector)
            return [result]

    def _get_many(self, collection_name: str, id):
        results = self._client.retrieve(
            collection_name, id, with_vectors=True, with_payload=False
        )
        if results:
            result = [np.array(result.vector) for result in results]
            return result

    def _parse_filter(self, filters: dict):
        return Filter(
            should=[
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
                for key, value in filters.items()
            ]
        )

    def _parse_distance(self, distance: Distance):
        match = {
            Distance.COSINE: QdrantDistance.COSINE,
            Distance.DOT: QdrantDistance.DOT,
        }

        qdrant_distance = match.get(distance, None)

        if not qdrant_distance:
            logging.warning(
                f"Distance '{distance}' not supported, setting distance to '{Distance.COSINE}'."
            )
            qdrant_distance = QdrantDistance.COSINE

        return qdrant_distance
