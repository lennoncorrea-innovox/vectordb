import logging
from typing import List, Dict, Any, Union
import numpy as np
from uuid import uuid4
from pymilvus import MilvusClient, DataType
from .vector_db import Distance, VectorDB


class Milvus(VectorDB):
    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.client = MilvusClient(f"http://{host}:{port}")

    def try_create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: Distance = Distance.COSINE,
    ):
        """Create a collection in Milvus."""
        if distance == Distance.COSINE:
            index_type = "IVF_FLAT"
            metric_type = "COSINE"
        elif distance == Distance.DOT:
            index_type = "IVF_FLAT"
            metric_type = "IP"
        else:
            raise ValueError("Unsupported distance")

        if self.client.has_collection(collection_name):
            logging.warning(f"Collection '{collection_name}' already exists.")
            return

        # Define the schema for the collection
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=36
        )
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_idx",
            index_type=index_type,
            metric_type=metric_type,
            params={"nlist": 128},
        )

        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            id_type="str",
            schema=schema,
            metric_type=metric_type,
            index_params=index_params,
            # consistency_level="Strong",
        )

        self.client.create_index(
            collection_name=collection_name, index_params=index_params, sync=True
        )

        logging.info(
            f"Collection '{collection_name}' created with dimension {dimension} and distance {distance}."
        )

    def add(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]] = None,
    ):
        """Add vectors and optional payloads to a collection."""
        try:
            self.client.insert(
                collection_name,
                [{"id": str(uuid4()), "vector": vector} for vector in vectors],
            )
            logging.info(
                f"Added {len(vectors)} vectors to collection '{collection_name}'."
            )
        except Exception as e:
            logging.error(f"Error inserting vectors: {e}")

    def get_by_id(self, collection_name: str, id: Union[str, list[str]]):
        if isinstance(id, list):
            result = self._get_many(collection_name, id)
        else:
            result = self.client.get(collection_name, id)
            if result:
                result = result[0]["vector"]
                result = np.array(result)

        return result

    def search(
        self,
        collection_name: str,
        query: Union[List[float], List[List[float]], np.ndarray],
        top_k: int = 5,
    ):
        """Search for similar vectors in a collection."""
        if isinstance(query, np.ndarray):
            if query.ndim == 1:  # Single vector, 1D
                query = [query.tolist()]
            elif query.ndim == 2:
                query = query.tolist()

        try:
            results = self.client.search(
                collection_name=collection_name,
                data=query,
                anns_field="vector",
                limit=top_k,
                # search_params={"metric_type": "COSINE"},
                output_fields=["id"],
            )
            return [[r["id"] for r in result] for result in results], [[r["score"] for r in result] for result in results]

        except Exception as e:
            logging.error(f"Error searching vectors: {e}")
            return []

    def update(
        self,
        collection_name: str,
        id: str,
        vector: Union[List[float], np.ndarray],
        payload: Dict[str, Any] = None,
    ):
        if isinstance(vector, np.ndarray):
            if vector.ndim == 2:
                vector = vector.squeeze(0)

        try:
            # In Milvus, updating involves deleting and re-inserting
            self.client.upsert(collection_name, {"id": id, "vector": vector})
            logging.info(
                f"Updated vector with id {id} in collection '{collection_name}'."
            )
        except Exception as e:
            logging.error(f"Error updating vector: {e}")

    def delete(self, collection_name: str, id: str):
        """Delete a vector by its point ID."""
        try:
            self.client.delete(collection_name, id)
            logging.info(
                f"Deleted vector with id {id} from collection '{collection_name}'."
            )
        except Exception as e:
            logging.error(f"Error deleting vector: {e}")

    def delete_collection(self, collection_name: str):
        """Delete a collection from Milvus."""
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logging.info(f"Deleted collection '{collection_name}'.")
        else:
            logging.warning(f"No collection named '{collection_name}' found.")

    # TODO: Check if statement if it happens
    def _get_many(self, collection_name: str, id: list[str]):
        results = self.client.get(collection_name, id)
        if results:
            result = np.array([np.array(result["vector"]) for result in results])
            return result

    # TODO: Check if statement if it happens
    def _get(self, collection_name: str, id: str):
        result = self.client.get(collection_name, id)
        if result:
            result = np.array(result[0]["vector"])
            return result
