"""
Tool for managing memories using memstore (store, retrieve, and search)
This module provides memory management capabilities using the custom memstore
implementation with FAISS and Ollama embeddings.

Key Features:
- store: Add new memories with userid and metadata
- retrieve: Perform semantic search across memories
- get_user_memories: Get all memories for a specific user
- search: Alias for retrieve with additional filtering options
"""

import json
import logging
from typing import Any, Dict, List, Optional
from memstore import memstore
from strands import Agent, tool
from strands.models.ollama import OllamaModel

logger = logging.getLogger(__name__)

TOOL_SPEC = {
    "name": "memstore_tool",
    "description": (
        "Memory management tool for storing and retrieving memories using FAISS and Ollama embeddings.\n\n"
        "Features:\n"
        "1. Store memories with userid and metadata\n"
        "2. Retrieve memories by semantic search\n"
        "3. Get all memories for a specific user\n"
        "4. Filter memories by metadata\n\n"
        "Actions:\n"
        "- store: Store new memory (requires userid, content)\n"
        "- retrieve: Semantic search across all memories\n"
        "- get_user: Get all memories for a specific user\n"
        "- search: Advanced search with metadata filtering\n"
        "- init: Initialize a new collection\n\n"
        "Note: The tool uses Ollama embeddings and FAISS for vector similarity search."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform (store, retrieve, get_user, search, init)",
                    "enum": ["store", "retrieve", "get_user", "search", "init"],
                },
                "collection_name": {
                    "type": "string",
                    "description": "Name of the memory collection (default: 'default')",
                },
                "userid": {
                    "type": "string",
                    "description": "User ID for memory operations (required for store and get_user)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to store (required for store action)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (required for retrieve and search actions)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to store with the memory or filter by",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return for search (default: 5)",
                },
            },
            "required": ["action"],
        }
    },
}


class MemstoreClient:
    """Client for interacting with memstore service."""

    def __init__(self):
        """Initialize the memstore client."""
        self.collections = {}

    def get_collection(self, collection_name: str = "default") -> memstore:
        """Get or create a memstore collection."""
        if collection_name not in self.collections:
            store = memstore(collection_name)
            try:
                store.load()
            except:
                store.create_collection()
            self.collections[collection_name] = store
        return self.collections[collection_name]

    def store_memory(
        self,
        collection_name: str,
        userid: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Store a memory in the collection."""
        if not userid:
            raise ValueError("userid is required for store action")
        if not content:
            raise ValueError("content is required for store action")

        store = self.get_collection(collection_name)
        store.insert(userid, content, metadata or {})

        return {
            "status": "success",
            "message": "Memory stored successfully",
            "userid": userid,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "metadata": metadata or {},
        }

    def retrieve_memories(
        self,
        collection_name: str,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """Retrieve memories using semantic search."""
        if not query:
            raise ValueError("query is required for retrieve action")

        store = self.get_collection(collection_name)
        results = store.get_relevant_memories(query, k, metadata_filter)

        return [
            {
                "score": result["score"],
                "userid": result["data"]["userid"],
                "content": result["data"]["content"],
                "metadata": result["data"]["metadata"],
            }
            for result in results
        ]

    def get_user_memories(
        self,
        collection_name: str,
        userid: str,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """Get all memories for a specific user."""
        if not userid:
            raise ValueError("userid is required for get_user action")

        store = self.get_collection(collection_name)
        results = store.get_memories_for_user(userid, metadata_filter)
        return [
            {
                "userid": result["data"]["userid"],
                "content": result["data"]["content"],
                "metadata": result["data"]["metadata"],
            }
            for result in results
        ]

    def init_collection(self, collection_name: str) -> Dict:
        """Initialize a new collection."""
        store = memstore(collection_name)
        store.create_collection()
        self.collections[collection_name] = store
        return {
            "status": "success",
            "message": f"Collection '{collection_name}' initialized successfully",
            "collection_name": collection_name,
        }

@tool
def memstore_tool(action: str, collection_name: str = "default", userid: str = None, content: str = None, query: str = None, metadata: dict = None, k: int = 5):
    """
    Memory management tool for storing and retrieving memories using memstore.
    
    Args:
        action: Action to perform (store, retrieve, get_user, search, init)
        collection_name: Name of the memory collection
        userid: User ID for memory operations
        content: Content to store (for store action)
        query: Search query (for retrieve/search actions)
        metadata: Optional metadata
        k: Number of results to return (default: 5)
    """
    try:
        client = MemstoreClient()
        
        if action == "store":
            return client.store_memory(collection_name, userid, content, metadata)
        elif action in ["retrieve", "search"]:
            return client.retrieve_memories(collection_name, query, k, metadata)
        elif action == "get_user":
            return client.get_user_memories(collection_name, userid, metadata)
        elif action == "init":
            return client.init_collection(collection_name)
        else:
            raise ValueError(f"Invalid action: {action}")
            
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage for testing
if __name__ == "__main__":
    ollama_model = OllamaModel(model_id="llama3.2:3b", host="http://localhost:11434")

    agent = Agent(name="memstore_agent", model=ollama_model, tools=[memstore_tool])
    #response = agent("From the collection agent_collection, store a memory for user Raja with content 'Raja works for AWS'")
    #print("Agent response:", response)
    response = agent("From the collection agent_collection, get all the memories for userid Raja")
    print("Agent response:", response)

