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
from strands import tool
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

logger = logging.getLogger(__name__)

TOOL_SPEC = {
    "name": "memstore_memory",
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
            "metadata": metadata or {}
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
                "metadata": result["data"]["metadata"]
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
                "metadata": result["data"]["metadata"]
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
            "collection_name": collection_name
        }

def format_store_response(result: Dict) -> str:
    """Format store memory response."""
    return (
        f"✅ Memory stored successfully:\n"
        f"👤 User ID: {result['userid']}\n"
        f"📄 Content Preview: {result['content_preview']}\n"
        f"📋 Metadata: {json.dumps(result['metadata'], indent=2) if result['metadata'] else 'None'}"
    )

def format_retrieve_response(memories: List[Dict]) -> str:
    """Format retrieve memories response."""
    if not memories:
        return "No memories found matching the query."
    
    result = f"🔍 Found {len(memories)} matching memories:\n\n"
    for i, memory in enumerate(memories, 1):
        content_preview = memory['content'][:150] + "..." if len(memory['content']) > 150 else memory['content']
        result += (
            f"{i}. Score: {memory['score']:.3f}\n"
            f"   👤 User: {memory['userid']}\n"
            f"   📄 Content: {content_preview}\n"
            f"   📋 Metadata: {json.dumps(memory['metadata']) if memory['metadata'] else 'None'}\n\n"
        )
    
    return {"text": result.strip()}

def format_user_memories_response(memories: List[Dict], userid: str) -> str:
    """Format user memories response."""
    if not memories:
        return f"No memories found for user: {userid}"
    
    result = f"👤 Found {len(memories)} memories for user '{userid}':\n\n"
    for i, memory in enumerate(memories, 1):
        content_preview = memory['content'][:150] + "..." if len(memory['content']) > 150 else memory['content']
        result += (
            f"{i}. 📄 Content: {content_preview}\n"
            f"   📋 Metadata: {json.dumps(memory['metadata']) if memory['metadata'] else 'None'}\n\n"
        )
    
    return {"text": result.strip()}

@tool
def memstore_memory(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Memory management tool for storing and retrieving memories using memstore.
    
    Args:
        tool_input: Dictionary containing:
            - action:\n"
                    "- store: Store new memory (requires userid, content)\n"
                    "- retrieve: Semantic search across all memories\n"
                    "- get_user: Get all memories for a specific user\n"
                    "- search: Advanced search with metadata filtering\n"
                    "- init: Initialize a new collection\n\n"
            - collection_name: Name of the memory collection (optional)
            - userid: User ID for memory operations
            - content: Content to store (for store action)
            - query: Search query (for retrieve/search actions)
            - metadata: Optional metadata
            - k: Number of results to return (default: 5)
    
    Returns:
        Dictionary containing status and response content
    """
    try:
        # Validate required parameters
        if not tool_input.get("action"):
            raise ValueError("action parameter is required")
        
        # Initialize client
        client = MemstoreClient()
        
        # Get parameters
        action = tool_input["action"]
        collection_name = tool_input.get("collection_name", "default")
        
        # Execute the requested action
        if action == "store":
            result = client.store_memory(
                collection_name,
                tool_input.get("userid"),
                tool_input.get("content"),
                tool_input.get("metadata"),
            )
            response_text = format_store_response(result)
            return {
                "status": "success",
                "content": [response_text],
            }
        
        elif action == "retrieve" or action == "search":
            memories = client.retrieve_memories(
                collection_name,
                tool_input.get("query"),
                tool_input.get("k", 5),
                tool_input.get("metadata"),
            )
            response_text = format_retrieve_response(memories)
            return {
                "status": "success",
                "content": [response_text],
            }
        
        elif action == "get_user":
            memories = client.get_user_memories(
                collection_name,
                tool_input.get("userid"),
                tool_input.get("metadata"),
            )
            response_text = format_user_memories_response(memories, tool_input.get("userid"))
            return {
                "status": "success",
                "content": [response_text],
            }
        
        elif action == "init":
            result = client.init_collection(collection_name)
            return {
                "status": "success",
                "content": [f"✅ {result['message']}"],
            }
        
        else:
            raise ValueError(f"Invalid action: {action}")
    
    except Exception as e:
        logger.error(f"Memstore tool error: {str(e)}")
        return {
            "status": "error",
            "content": [f"Error: {str(e)}"],
        }

# Example usage for testing
if __name__ == "__main__":
    # Test init
    result = memstore_memory({
        "action": "init",
        "collection_name": "test_collection"
    })
    print("Init result:", result)
    
    # Test store
    result = memstore_memory({
        "action": "store",
        "collection_name": "test_collection",
        "userid": "test_user",
        "content": "This is a test memory about machine learning concepts",
        "metadata": {"category": "learning", "topic": "ml"}
    })
    print("Store result:", result)
    
    # Test retrieve
    result = memstore_memory({
        "action": "retrieve",
        "collection_name": "test_collection",
        "query": "machine learning",
        "k": 3
    })
    print("Retrieve result:", result)
    
    # Test get_user
    result = memstore_memory({
        "action": "get_user",
        "collection_name": "test_collection",
        "userid": "test_user"
    })
    print("Get user result:", result)


    bedrock_model = BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        region_name="us-west-2"
    )

    agent = Agent(name="memstore_agent", 
                  model=bedrock_model,
                  tools=[memstore_memory])
    response = agent( "Where is Raja living according to the collection agent_test_collection" )
    print("Agent response:", response)