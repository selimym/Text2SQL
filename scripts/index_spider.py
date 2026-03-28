"""CLI script to index Spider training data into ChromaDB."""

import argparse
import asyncio
import json
from pathlib import Path

import chromadb
import sqlglot

from app.core.config import get_settings
from app.db.spider_loader import load_schema_documents_from_spider
from app.retrieval.embeddings import get_embeddings
from app.retrieval.example_doc import ExampleDocument
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever


def extract_tables_from_sql(sql: str) -> list[str]:
    """Extract table names referenced in a SQL query using sqlglot."""
    try:
        tables: list[str] = []
        for expr in sqlglot.parse(sql):
            if expr is not None:
                for table in expr.find_all(sqlglot.exp.Table):
                    if table.name:
                        tables.append(table.name.lower())
        return list(set(tables))
    except Exception:
        return []


async def main() -> None:
    parser = argparse.ArgumentParser(description="Index Spider data into ChromaDB")
    parser.add_argument(
        "--spider-dir",
        default="spider_data",
        help="Path to Spider dataset directory",
    )
    args = parser.parse_args()

    spider_dir = Path(args.spider_dir)
    settings = get_settings()

    print(f"Using Spider data from: {spider_dir}")
    print(f"ChromaDB persist dir: {settings.chroma_persist_dir}")

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embeddings = get_embeddings(settings.embedding_provider, settings.embedding_model)

    # Index schema documents
    tables_json_path = str(spider_dir / "tables.json")
    print(f"\nLoading schema documents from {tables_json_path}...")
    schema_docs = load_schema_documents_from_spider(tables_json_path)
    print(f"Loaded {len(schema_docs)} schema documents")

    schema_retriever = SchemaRetriever(
        embeddings=embeddings,
        collection_name="spider_schemas",
        chroma_client=chroma_client,
    )
    await schema_retriever.index(schema_docs)
    print(f"Indexed {len(schema_docs)} schema documents into ChromaDB")

    # Index example documents
    train_json_path = spider_dir / "train_spider.json"
    print(f"\nLoading training examples from {train_json_path}...")
    with open(train_json_path) as f:
        train_data = json.load(f)

    example_docs: list[ExampleDocument] = []
    for item in train_data:
        tables_used = extract_tables_from_sql(item["query"])
        example_docs.append(
            ExampleDocument(
                db_id=item["db_id"],
                question=item["question"],
                sql=item["query"],
                tables_used=tables_used,
                difficulty=item.get("hardness"),
            )
        )

    print(f"Loaded {len(example_docs)} training examples")

    example_retriever = ExampleRetriever(
        embeddings=embeddings,
        collection_name="spider_examples",
        chroma_client=chroma_client,
    )
    await example_retriever.index(example_docs)
    print(f"Indexed {len(example_docs)} training examples into ChromaDB")
    print("\nIndexing complete!")


if __name__ == "__main__":
    asyncio.run(main())
