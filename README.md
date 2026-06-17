# RAG Drug System

A Retrieval-Augmented Generation (RAG) system for drug-related queries.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the application:

```bash
uvicorn app:app --reload
```

3. Access the API:

```
http://127.0.0.1:8000/ask?query=your+question
```

## API Endpoints

- **GET `/ask`** - Submit a query and receive a response from the RAG pipeline
  - Query parameter: `query` (string)
  - Response: `{"response": "..."}`

## Project Structure

- `app.py` - FastAPI application entry point
- `pipeline.py` - Main RAG pipeline logic
- `retriever.py` - Document retrieval module
- `embeddings.py` - Embedding generation
- `llm.py` - Language model interface
- `db.py` - Database connection
- `db_api.py` - Database API
- `config.py` - Configuration settings
