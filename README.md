# QFlow - Agentic SDLC Implementation

An agentic implementation for Software Development Life Cycle (SDLC) with integrated memory management using FAISS vector database and Ollama embeddings.

## Features

- **Vector Memory Store**: FAISS-based vector database for efficient similarity search
- **Ollama Integration**: Uses Ollama embeddings (llama3.2:3b) for semantic understanding
- **User-based Memory**: Store and retrieve memories by user ID
- **Metadata Filtering**: Advanced search with metadata criteria
- **Strands-Agents Tool**: Ready-to-use tool for agent integration

## Installation

```bash
uv sync
# or
pip install -e .
```

## Usage

### Direct memstore Usage

```python
from memstore.memstore import memstore

# Create and initialize a collection
store = memstore("my_collection")
store.create_collection()

# Insert memories
store.insert(
    userid='user123', 
    content='I love sushi for dinner', 
    metadata={'category': 'food'}
)

# Semantic search
results = store.get_relevant_memories("favorite food", k=5)

# Get user-specific memories
user_memories = store.get_memories_for_user("user123")

# Load existing collection
loaded_store = memstore("my_collection")
loaded_store.load()
```

### Strands-Agents Tool Integration

```python
from memstore.memstore_tool import memstore_tool
from strands import Agent
from strands.models.bedrock import BedrockModel

model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
agent = Agent(name="memory_agent", model=model, tools=[memstore_tool])

response = agent("Store a memory: I prefer working in the morning")
```

## Core Components

### memstore Class

- **Vector Database**: FAISS IndexFlatIP for inner product similarity
- **Embedding Model**: Ollama llama3.2:3b (3072 dimensions)
- **Persistence**: Automatic save/load with .faiss and .pkl files
- **Methods**:
  - `create_collection()`: Initialize new collection
  - `insert(userid, content, metadata)`: Add new memory
  - `get_relevant_memories(query, k, metadata_filter)`: Semantic search
  - `get_memories_for_user(userid, metadata_filter)`: User-specific retrieval
  - `load()`: Load existing collection

### memstore_tool

Strands-agents compatible tool with actions:
- `store`: Add new memory
- `retrieve`/`search`: Semantic search
- `get_user`: Get user memories
- `init`: Initialize collection

## Project Structure

```
qflow/
├── memstore/
│   ├── memstore.py          # Core FAISS vector store
│   ├── memstore_tool.py     # Strands-agents tool wrapper
│   └── memstore_client.ipynb # Usage examples
├── repo/                    # FAISS index storage
├── pyproject.toml          # Project dependencies
└── README.md
```

## Dependencies

- `faiss-cpu>=1.11.0` - Vector similarity search
- `ollama>=0.5.1` - Embedding generation
- `strands-agents>=0.1.9` - Agent framework
- `numpy>=2.3.1` - Numerical operations

## Example Notebook

See `memstore/memstore_client.ipynb` for complete usage examples including:
- Collection creation and loading
- Memory insertion and retrieval
- Agent integration with strands-agents