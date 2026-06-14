#!/usr/bin/env python3
"""Build a ChromaDB vector knowledge base from PDFs in the docs directory."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import warnings
from pathlib import Path
from typing import Iterable

warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback for this project's current virtualenv. Prefer installing
    # langchain-chroma for new environments.
    from langchain_community.vectorstores import Chroma


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs"
DEFAULT_DB_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION = "pdf_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read PDFs from docs/ and save OpenAI embeddings to ChromaDB."
    )
    parser.add_argument(
        "--docs-dir",
        default=str(DEFAULT_DOCS_DIR),
        help="Directory containing PDF files. Default: ./docs",
    )
    parser.add_argument(
        "--db-dir",
        default=str(DEFAULT_DB_DIR),
        help="ChromaDB persistence directory. Default: ./chroma_db",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection name. Default: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help=f"OpenAI embedding model. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per text chunk. Default: 1000",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Characters shared between neighboring chunks. Default: 150",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks to add to ChromaDB per batch. Default: 100",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep the existing ChromaDB directory instead of rebuilding it.",
    )
    return parser.parse_args()


def find_pdfs(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {docs_dir}")
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"Docs path is not a directory: {docs_dir}")
    return sorted(p for p in docs_dir.rglob("*.pdf") if p.is_file())


def sanitize_metadata(metadata: dict) -> dict:
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def load_pdf_documents(pdf_paths: Iterable[Path], docs_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        for document in loader.load():
            rel_source = pdf_path.relative_to(docs_dir).as_posix()
            document.metadata = sanitize_metadata(document.metadata)
            document.metadata.update(
                {
                    "source": rel_source,
                    "file_name": pdf_path.name,
                    "absolute_path": str(pdf_path),
                }
            )
            if document.page_content.strip():
                documents.append(document)
    return documents


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。", "！", "？", "；",
            ". ", "! ", "? ", "; ",
            "，", ", ",
            " ",
            "",
        ],
    )

    chunks: list[Document] = []
    for document in documents:
        page_chunks = splitter.split_documents([document])
        for index, chunk in enumerate(page_chunks):
            chunk.page_content = chunk.page_content.strip()
            if not chunk.page_content:
                continue
            chunk.metadata = sanitize_metadata(chunk.metadata)
            chunk.metadata["chunk_index"] = index
            chunks.append(chunk)
    return chunks


def stable_chunk_id(chunk: Document) -> str:
    source = str(chunk.metadata.get("source", "unknown"))
    page = str(chunk.metadata.get("page", "unknown"))
    chunk_index = str(chunk.metadata.get("chunk_index", "unknown"))
    content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{source}:page-{page}:chunk-{chunk_index}:{content_hash}"


def batch_items(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def assert_safe_to_reset(db_dir: Path) -> None:
    resolved = db_dir.resolve()
    protected = {PROJECT_ROOT, PROJECT_ROOT.parent, Path.home(), Path("/")}
    if resolved in protected or len(resolved.parts) < 3:
        raise ValueError(f"Refusing to delete unsafe db directory: {resolved}")


def reset_db_dir(db_dir: Path) -> None:
    if not db_dir.exists():
        return
    assert_safe_to_reset(db_dir)
    shutil.rmtree(db_dir)


def build_vector_store(
    chunks: list[Document],
    db_dir: Path,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
) -> Chroma:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(db_dir),
    )

    ids = [stable_chunk_id(chunk) for chunk in chunks]
    for chunk_batch, id_batch in zip(
        batch_items(chunks, batch_size), batch_items(ids, batch_size), strict=True
    ):
        vector_store.add_documents(chunk_batch, ids=id_batch)

    persist = getattr(vector_store, "persist", None)
    if callable(persist):
        persist()

    return vector_store


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env or export it before running."
        )

    docs_dir = Path(args.docs_dir).expanduser().resolve()
    db_dir = Path(args.db_dir).expanduser().resolve()

    pdf_paths = find_pdfs(docs_dir)
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found under: {docs_dir}")

    if not args.keep_existing:
        reset_db_dir(db_dir)

    print(f"Found {len(pdf_paths)} PDF file(s).")
    raw_documents = load_pdf_documents(pdf_paths, docs_dir)
    if not raw_documents:
        raise RuntimeError("No readable text was extracted from the PDF files.")

    chunks = split_documents(raw_documents, args.chunk_size, args.chunk_overlap)
    if not chunks:
        raise RuntimeError("No text chunks were created from the PDF files.")

    print(f"Loaded {len(raw_documents)} page document(s).")
    print(f"Created {len(chunks)} text chunk(s).")
    print(f"Building ChromaDB collection '{args.collection_name}' in {db_dir} ...")

    vector_store = build_vector_store(
        chunks=chunks,
        db_dir=db_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )

    try:
        count = vector_store._collection.count()
        print(f"ChromaDB saved. Collection contains {count} item(s).")
    except Exception:
        print("ChromaDB saved.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
