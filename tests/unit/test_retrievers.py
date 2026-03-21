import uuid
from unittest.mock import MagicMock

import chromadb
import pytest

from app.db.schema_doc import ColumnInfo, SchemaDocument
from app.retrieval.example_doc import ExampleDocument
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever


def make_schema_doc(db_id: str, table_name: str) -> SchemaDocument:
    return SchemaDocument(
        db_id=db_id,
        table_name=table_name,
        columns=[ColumnInfo(name="id", data_type="INTEGER", is_primary_key=True)],
        foreign_keys=[],
    )


@pytest.fixture
def mock_embeddings() -> MagicMock:
    mock = MagicMock()
    # Return fixed-length vectors for any text
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 10
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def schema_retriever(mock_embeddings: MagicMock) -> SchemaRetriever:
    client = chromadb.EphemeralClient()
    # Use a unique collection name per test to avoid cross-test state leakage
    # (EphemeralClient instances share the same in-memory backend)
    collection_name = f"test_schema_{uuid.uuid4().hex}"
    return SchemaRetriever(
        embeddings=mock_embeddings,
        collection_name=collection_name,
        chroma_client=client,
    )


def test_index_and_retrieve(schema_retriever: SchemaRetriever) -> None:
    docs = [
        make_schema_doc("db1", "singer"),
        make_schema_doc("db1", "concert"),
        make_schema_doc("db2", "other_table"),
    ]
    schema_retriever.index(docs)
    results = schema_retriever.retrieve("How many singers?", db_id="db1", top_k=2)
    assert len(results) == 2
    result_tables = {r.table_name for r in results}
    assert result_tables.issubset({"singer", "concert"})


def test_db_id_filter(schema_retriever: SchemaRetriever) -> None:
    docs = [
        make_schema_doc("db1", "singer"),
        make_schema_doc("db2", "other_table"),
    ]
    schema_retriever.index(docs)
    results = schema_retriever.retrieve("test query", db_id="db1", top_k=5)
    assert all(r.db_id == "db1" for r in results)
    assert len(results) == 1


def make_example_doc(db_id: str, question: str, sql: str) -> ExampleDocument:
    return ExampleDocument(db_id=db_id, question=question, sql=sql)


@pytest.fixture
def example_retriever(mock_embeddings: MagicMock) -> ExampleRetriever:
    client = chromadb.EphemeralClient()
    collection_name = f"test_examples_{uuid.uuid4().hex}"
    return ExampleRetriever(
        embeddings=mock_embeddings,
        collection_name=collection_name,
        chroma_client=client,
    )


def test_example_index_and_retrieve(example_retriever: ExampleRetriever) -> None:
    docs = [
        make_example_doc("db1", "How many singers?", "SELECT COUNT(*) FROM singer"),
        make_example_doc("db1", "List all concerts", "SELECT * FROM concert"),
        make_example_doc("db1", "Find singers by name", "SELECT * FROM singer WHERE name = 'x'"),
        make_example_doc("db2", "Other db question", "SELECT * FROM other"),
        make_example_doc("db2", "Another question", "SELECT id FROM other"),
    ]
    example_retriever.index(docs)
    results = example_retriever.retrieve("How many rows?", db_id=None, top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, ExampleDocument) for r in results)


def test_example_deserialized_correctly(example_retriever: ExampleRetriever) -> None:
    docs = [make_example_doc("db1", "How many singers?", "SELECT COUNT(*) FROM singer")]
    example_retriever.index(docs)
    results = example_retriever.retrieve("singers count", db_id="db1", top_k=1)
    assert len(results) == 1
    assert results[0].question == "How many singers?"
    assert results[0].sql == "SELECT COUNT(*) FROM singer"
    assert results[0].db_id == "db1"
