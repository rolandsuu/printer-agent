#!/usr/bin/env python3
"""Query the local ChromaDB knowledge base built by build_rag.py."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "pdf_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Error: OPENAI_API_KEY is not set in .env or the shell.")

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )

    query = "怎么安装打印头"
    results = db.similarity_search(query, k=3)

    if not results:
        print("No matching documents found.")
        return

    for index, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", "unknown page")
        print(f"\n--- Result {index}: {source}, page {page} ---")
        print(doc.page_content)


if __name__ == "__main__":
    main()
