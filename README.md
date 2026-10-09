# Document Q&A Assistant

A RAG (Retrieval-Augmented Generation) application that lets you upload 
a text document and ask questions about its content, using semantic 
search and a language model to generate answers.

## Features

- Upload text documents and store them as searchable chunks
- Ask questions and get answers generated from relevant document content
- Persistent storage — uploaded documents remain available across restarts
- View all uploaded documents and clear the database
- Simple web interface for interacting with the system
- Automated test suite covering core functionality

## Tech stack

- **FastAPI** — web framework
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — text embeddings
- **ChromaDB** — persistent vector database
- **Transformers** (`google/flan-t5-small`) — answer generation
- **pytest** — automated testing

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` in a browser for the web interface, or 
`http://127.0.0.1:8000/docs` for the interactive API documentation.

## Running tests

```bash
pytest test_main.py -v
```

## API Endpoints

| Method | Endpoint     | Description                          |
|--------|--------------|---------------------------------------|
| GET    | `/`          | Web interface                         |
| GET    | `/health`    | Health check                          |
| POST   | `/upload`    | Upload a text document                |
| POST   | `/query`     | Ask a question about uploaded docs    |
| GET    | `/documents` | List all uploaded documents           |
| DELETE | `/clear`     | Clear all documents from the database |