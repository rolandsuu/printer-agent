#!/usr/bin/env python3
"""Chat with the local PDF knowledge base using retrieved manual context."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "pdf_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """你是一个专业、耐心的打印机说明书助手。

回答规则：
- 只根据用户问题和参考资料回答，不要编造。
- 如果参考资料不够，就直接说“我在说明书里没有找到足够信息”。
- 用简洁中文回答，语气像 ChatGPT 一样自然。
- 如果是操作问题，用编号步骤。
- 如果有安全风险或注意事项，要提醒用户。
- 最后加一个“参考来源”小节，列出你用到的文件和页码。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the PDF knowledge base and get a ChatGPT-like answer."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. If omitted, an interactive chat starts.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of matching chunks to use as context. Default: 4",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Answer creativity. Lower is more factual. Default: 0.2",
    )
    return parser.parse_args()


def require_environment() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Error: OPENAI_API_KEY is not set in .env or the shell.")
    if not DB_DIR.exists():
        raise SystemExit(
            "Error: chroma_db/ was not found. Run `python build_rag.py` first."
        )


def create_vector_store() -> Chroma:
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=OpenAIEmbeddings(model=embedding_model),
    )


def create_chat_model(temperature: float) -> ChatOpenAI:
    chat_model = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    return ChatOpenAI(model=chat_model, temperature=temperature)


def page_label(doc: Document) -> str:
    page = doc.metadata.get("page")
    if page is None:
        return "未知页码"
    try:
        return f"第{int(page) + 1}页"
    except (TypeError, ValueError):
        return f"第{page}页"


def source_label(doc: Document) -> str:
    source = (
        doc.metadata.get("source")
        or doc.metadata.get("file_name")
        or "unknown source"
    )
    return f"{source}, {page_label(doc)}"


def format_context(documents: list[Document]) -> str:
    parts = []
    for index, doc in enumerate(documents, start=1):
        parts.append(
            f"[参考资料 {index}: {source_label(doc)}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(parts)


def build_messages(question: str, documents: list[Document]) -> list[BaseMessage]:
    user_prompt = f"""用户问题：
{question}

参考资料：
{format_context(documents)}
"""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]


def unique_sources(documents: list[Document]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for doc in documents:
        label = source_label(doc)
        if label in seen:
            continue
        seen.add(label)
        lines.append(f"- {label}")
    return "\n".join(lines)


def ask_question(db: Chroma, llm: ChatOpenAI, question: str, k: int) -> str:
    documents = db.similarity_search(question, k=k)
    if not documents:
        return "我在说明书里没有找到相关内容。"

    response = llm.invoke(build_messages(question, documents))
    answer = str(response.content).strip()

    if "参考来源" not in answer:
        answer = f"{answer}\n\n参考来源：\n{unique_sources(documents)}"
    return answer


def run_once(db: Chroma, llm: ChatOpenAI, question: str, k: int) -> None:
    print("\n助手：")
    print(ask_question(db, llm, question, k))


def run_interactive(db: Chroma, llm: ChatOpenAI, k: int) -> None:
    print("Printer Agent chat is ready. Type q, quit, or exit to stop.\n")
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if question.lower() in {"q", "quit", "exit"}:
            return
        if not question:
            continue

        run_once(db, llm, question, k)
        print()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    if args.k <= 0:
        raise SystemExit("Error: --k must be greater than 0.")

    require_environment()
    db = create_vector_store()
    llm = create_chat_model(args.temperature)

    question = " ".join(args.question).strip()
    if question:
        run_once(db, llm, question, args.k)
    else:
        run_interactive(db, llm, args.k)


if __name__ == "__main__":
    main()
