import ollama
import os
import numpy as np
import pickle
import logging

try:
    logging.getLogger("faiss").setLevel(logging.INFO)
    logging.getLogger("faiss.loader").setLevel(logging.INFO)
    import faiss
except ImportError:
    raise ImportError(
        "Could not import faiss python package. "
        "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
        "or `pip install faiss-cpu` (depending on Python version)."
    )

logger = logging.getLogger(__name__)


class memstore:
    """A vector database implementation using FAISS for similarity search with Ollama embeddings.
    
    This class provides functionality to store, retrieve, and manage text embeddings
    with associated metadata using FAISS indexing and Ollama's embedding models.
    """
    
    def __init__(self, collection_name):
        """Initialize a new memstore instance.
        
        Args:
            collection_name (str): Name of the collection to create or load.
        """
        self.collection_name = collection_name
        self.embedding_model_dims = 3072
        self.embedding_model = "llama3.2:3b"
        self.collection_path = 'repo'
        self.index_path = f"repo/{collection_name}.faiss"
        self.metadata_store_path = f"repo/{collection_name}.pkl"
        self.metadata_store = {}  # {index: {"userid": str, "content": str, "metadata": dict}}
        self.userid_index = {}    # {userid: [indices]}
        self.index = None

    def create_collection(self):
        """Create a new FAISS collection with Inner Product index and save to disk."""
        self.index = faiss.IndexFlatIP(self.embedding_model_dims)
        self._save()

    def _save(self):
        """Save the current FAISS index and metadata store to disk.
        
        Creates the collection directory if it doesn't exist and handles save errors gracefully.
        """
        try:
            if not os.path.exists(self.collection_path):
                os.makedirs(self.collection_path)
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_store_path, 'wb') as f:
                pickle.dump({"metadata_store": self.metadata_store, "userid_index": self.userid_index}, f)
        except Exception as e:
            logger.warning(f"Failed to save FAISS index : {e}")

    def _rebuild_userid_index(self):
        """Rebuild userid index from metadata store for backward compatibility."""
        self.userid_index = {}
        for idx, data in self.metadata_store.items():
            if isinstance(data, dict) and "userid" in data:
                userid = data["userid"]
                if userid not in self.userid_index:
                    self.userid_index[userid] = []
                self.userid_index[userid].append(idx)

    def get_relevant_memories(self, query, k=5, metadata_filter=None):
        """Search by content similarity using vector embeddings.
        
        Args:
            query (str): Query text to search for similar content.
            k (int): Number of results to return.
            metadata_filter (dict): Optional metadata filter criteria.
            
        Returns:
            list: List of matching results with scores.
        """
        emb = ollama.embed(model=self.embedding_model, input=query)
        query_vector = np.array(emb.embeddings, dtype=np.float32)
        
        scores, indices = self.index.search(query_vector, k)
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.metadata_store:
                data = self.metadata_store[idx]
                if self._matches_filter(data.get("metadata", {}), metadata_filter):
                    results.append({"score": float(score), "data": data})
        
        return results

    def get_memories_for_user(self, userid, metadata_filter=None):
        """Search by userid for direct retrieval.
        
        Args:
            userid (str): User identifier to search for.
            metadata_filter (dict): Optional metadata filter criteria.
            
        Returns:
            list: List of matching results for the userid.
        """
        if userid not in self.userid_index:
            return []
        
        results = []
        for idx in self.userid_index[userid]:
            if idx in self.metadata_store:
                data = self.metadata_store[idx]
                if self._matches_filter(data.get("metadata", {}), metadata_filter):
                    results.append({"data": data})
        
        return results

    def _matches_filter(self, metadata, filter_criteria):
        """Check if metadata matches filter criteria.
        
        Args:
            metadata (dict): Metadata to check.
            filter_criteria (dict): Filter criteria to match against.
            
        Returns:
            bool: True if metadata matches all filter criteria.
        """
        if not filter_criteria:
            return True
        
        for key, value in filter_criteria.items():
            if key not in metadata or metadata[key] != value:
                return False
        
        return True

    def insert(self, userid, content, metadata):
        """Embed content using Ollama and insert into the FAISS index with userid and metadata.
        
        Args:
            userid (str): User identifier for direct lookup.
            content (str): Text content to be embedded and stored.
            metadata (dict): Associated metadata to store with the embedding.
        """
        emb = ollama.embed(model=self.embedding_model, input=content)
        vector = np.array(emb.embeddings, dtype=np.float32)

        idx = len(self.metadata_store)
        self.index.add(vector)
        self.metadata_store[idx] = {"userid": userid, "content": content, "metadata": metadata}
        
        if userid not in self.userid_index:
            self.userid_index[userid] = []
        self.userid_index[userid].append(idx)
        self._save()

    def load(self):
        """Load an existing FAISS index and metadata store from disk.
        
        Handles loading errors gracefully by initializing empty metadata store on failure.
        """
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_store_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict) and "metadata_store" in data:
                self.metadata_store = data["metadata_store"]
                self.userid_index = data.get("userid_index", {})
            else:
                self.metadata_store = data
                self._rebuild_userid_index()
            logger.info(f"Loaded FAISS index from {self.index_path} with {len(self.metadata_store)} vectors")
        logger.info(f"Failed to load FAISS index from {self.index_path}")