import os
import logging
from pathlib import Path
from typing import Dict, List, TypedDict, Union
import numpy as np
import faiss
from .vector_db import Distance


class Index(TypedDict):
    index: faiss.Index
    dimension: int


class Faiss:
    def __init__(self):
        self.collections: Dict[str, Index] = {}
        self.save_path = self._get_collection_folder()
        self._load_collections_from_disk()
        
    def create_collection(
        self, collection_name: str, dimension: int, distance=Distance.EUCLID
    ):
        if collection_name in self.collections:
            logging.warning(f"Index for collection '{collection_name}' already exists.")
            return

        # Create a new FAISS index based on the provided type
        if distance == Distance.EUCLID:
            index = faiss.IndexFlatL2(dimension)
        elif distance == Distance.DOT:
            index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError("Unsupported distance")

        # Initialize the collection with the index and dimension
        self.collections[collection_name] = {"index": index, "dimension": dimension}
        logging.info(
            (
                f"Collection '{collection_name}' created with vector size {dimension} and distance {distance}."
            )
        )

    def add(self, collection_name: str, vectors: np.ndarray):
        if collection_name not in self.collections:
            raise ValueError(f"No index found for collection '{collection_name}'.")

        if vectors.shape[1] != self.collections[collection_name]["dimension"]:
            raise ValueError(
                f"Vectors must have {self.collections[collection_name]['dimension']} dimensions."
            )

        self.collections[collection_name]["index"].add(vectors)
        logging.info(f"Added {len(vectors)} vectors to collection '{collection_name}'.")

    def get(self, collection_name: str, id: Union[int, list[int]]):
        if id == 0 or id:
            if isinstance(id, list):
                if id[-1]-id[0]+1 == len(id):
                    return self._get_in_sequence(collection_name, id[0], len(id))
                result = []
                for idx in id:
                    result.append(self._get(collection_name, idx))
                return result
            else:
                return self._get(collection_name, id)
        return None

    def search(
        self,
        collection_name: str,
        query: Union[np.ndarray, list[np.ndarray]],
        top_k: int = 5,
    ):
        if isinstance(query, list):
            return self._search_vectors(collection_name, query, top_k)
        return self._search_vector(collection_name, query, top_k)

    def update(self, collection_name: str, id, vector: np.ndarray):
        logging.warning("FAISS does not support in-place vector updates.")
        pass
        # raise NotImplementedError("FAISS does not support in-place vector updates.")

    def delete(self, collection_name: str, id: int):
        logging.warning("FAISS does not support in-place vector deletes.")
        pass

    def delete_collection(self, collection_name: str):
        if collection_name in self.collections:
            filename = os.path.join(
                self.save_path,
                f"{collection_name}%{self.collections[collection_name]['dimension']}.index",
            )
            del self.collections[collection_name]
            if os.path.exists(filename):
                os.remove(filename)
            logging.info(f"Deleted collection '{collection_name}'.")

        else:
            logging.error(f"No index found for collection '{collection_name}'.")

    def save_collection(self, collection_name: str):
        if collection_name not in self.collections:
            logging.error(f"No index found for collection '{collection_name}'.")
            return

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        filename = os.path.join(
            self.save_path,
            f"{collection_name}%{self.collections[collection_name]['dimension']}.index",
        )

        faiss.write_index(self.collections[collection_name]["index"], filename)
        logging.info(f"Index for collection '{collection_name}' saved.")

    def load_collection(self, collection_name: str):
        filename = [
            filenames
            for filenames in os.listdir(self.save_path)
            if filenames.startswith(collection_name)
        ]
        if not filename:
            return
        filename = os.path.join(self.save_path, filename[0])

        if not os.path.exists(os.path.join(self.save_path, filename)):
            logging.error(f"No index found for collection '{collection_name}'.")
            return

        index = faiss.read_index(filename)
        _, dimension = self._get_collection_name_dim_from_filename(filename)
        self.collections[collection_name] = {
            "index": index,
            "dimension": int(dimension),
        }
        return index

    def _search_vector(self, collection_name: str, query: np.ndarray, top_k: int = 5):
        result = []

        if collection_name not in self.collections:
            logging.error(f"No index found for collection '{collection_name}'.")
            return result

        if query.shape[0] != self.collections[collection_name]["dimension"]:
            logging.error(
                f"Query vector must have {self.collections[collection_name]['dimension']} dimensions."
            )
            return result

        query = query.reshape(1, -1)  # Ensure the query vector is 2D
        distances_list, ids_list = self.collections[collection_name]["index"].search(
            query, top_k
        )
        return [list(ids) for ids in ids_list]

    def _search_vectors(self, collection_name: str, query: np.ndarray, top_k: int = 5):
        result = []
        for q in query:
            rn = self._search_vector(collection_name, q, top_k)
            if rn:
                result.append(rn[0])
        return result

    def _load_collections_from_disk(self):
        if os.path.exists(self.save_path):
            collection_names = os.listdir(self.save_path)
            for collection in collection_names:
                collection_name, dimension = tuple(collection.split("%"))
                dimension = dimension.split(".")[0]
                self.load_collection(collection_name)

    def _get_collection_folder(self):
        # Get the home directory of the user
        home_dir = Path.home()

        # Define the folder where collections will be stored, e.g., ~/my_collections
        collection_folder = home_dir / "my_collections"

        # Create the directory if it doesn't exist
        collection_folder.mkdir(parents=True, exist_ok=True)

        return collection_folder

    def _get_collection_name_dim_from_filename(self, filename: str):
        filename_splitted = filename.split("%")
        collection_name = filename_splitted[0]
        dimension = filename_splitted[1].split(".")[0]
        return collection_name, int(dimension)

    def _get(self, collection_name: str, id: int):
        return np.expand_dims(self.collections[collection_name]["index"].reconstruct(id), 0)
    
    def _get_in_sequence(self, collection_name, id: list[int], n):
        return self.collections[collection_name]["index"].reconstruct_n(id, n)
    