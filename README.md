# printer-agent

A small local RAG project that builds a Chroma vector database from PDF manuals and searches it with OpenAI embeddings.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env`.

## Build the knowledge base

Put PDF files in `docs/`, then run:

```bash
python build_rag.py
```

This creates a local `chroma_db/` folder. The folder is ignored by git because it is generated.

## Query the knowledge base

```bash
python test_rag.py
```
