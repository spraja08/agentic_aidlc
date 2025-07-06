# QFlow - Memory Store Tool

A memory management tool that converts the custom `memstore` implementation into a tool that any agent can use, following the strands-agents pattern.

## Features

- **Store memories** with user ID and metadata
- **Semantic search** across all memories using Ollama embeddings
- **User-specific retrieval** of all memories for a given user
- **Metadata filtering** for advanced search capabilities
- **FAISS vector database** for efficient similarity search

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Tool Usage

```python
from memstore_tool import memstore_memory

# Store a memory
result = memstore_memory({
    "action": "store",
    "userid": "user123",
    "content": "Important information to remember",
    "metadata": {"category": "notes", "priority": "high"}
})

# Search memories
result = memstore_memory({
    "action": "retrieve",
    "query": "important information",
    "k": 5
})

# Get all memories for a user
result = memstore_memory({
    "action": "get_user",
    "userid": "user123"
})
```

### Agent Integration

```python
from memstore_tool import memstore_memory, TOOL_SPEC

class MyAgent:
    def __init__(self):
        self.tools = {"memstore_memory": memstore_memory}
    
    def remember(self, content, metadata=None):
        return self.tools["memstore_memory"]({
            "action": "store",
            "userid": "my_agent",
            "content": content,
            "metadata": metadata or {}
        })
    
    def recall(self, query):
        return self.tools["memstore_memory"]({
            "action": "retrieve",
            "query": query
        })
```

## Tool Specification

- **Name**: `memstore_memory`
- **Actions**: 
  - `store`: Store new memory (requires userid, content)
  - `retrieve`: Semantic search across memories
  - `get_user`: Get all memories for a specific user
  - `search`: Advanced search with metadata filtering

## Files

- `memstore.py` - Original FAISS-based memory store implementation
- `memstore_tool.py` - Tool wrapper following strands-agents pattern
- `example_agent_usage.py` - Example demonstrating agent usage
- `requirements.txt` - Dependencies

## Example

Run the example to see the tool in action:

```bash
python example_agent_usage.py
```