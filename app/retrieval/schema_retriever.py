import re
from typing import Any, cast

from chromadb.api import ClientAPI
from langchain_core.embeddings import Embeddings

from app.db.schema_doc import ColumnInfo, ForeignKeyInfo, SchemaDocument


def _parse_schema_text(text: str, db_id: str, table_name: str) -> SchemaDocument:
    """Reconstruct a SchemaDocument from the stored to_text() format."""
    columns: list[ColumnInfo] = []
    foreign_keys: list[ForeignKeyInfo] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Columns:"):
            col_str = line[len("Columns:"):].strip()
            for part in col_str.split(","):
                part = part.strip()
                m = re.match(r"(\w+)\s+\(([^)]+)\)(\s+\[PK\])?", part)
                if m:
                    columns.append(
                        ColumnInfo(
                            name=m.group(1),
                            data_type=m.group(2),
                            is_primary_key=bool(m.group(3)),
                        )
                    )
        elif line.startswith("FK:"):
            m = re.match(r"FK:\s+(\w+)\s+->\s+(\w+)\.(\w+)", line)
            if m:
                foreign_keys.append(
                    ForeignKeyInfo(
                        from_column=m.group(1),
                        to_table=m.group(2),
                        to_column=m.group(3),
                    )
                )
    return SchemaDocument(
        db_id=db_id,
        table_name=table_name,
        columns=columns,
        foreign_keys=foreign_keys,
    )


class SchemaRetriever:
    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str,
        chroma_client: ClientAPI,
    ) -> None:
        self.embeddings = embeddings
        self.collection = chroma_client.get_or_create_collection(collection_name)

    def index(self, docs: list[SchemaDocument]) -> None:
        """Index schema documents into ChromaDB."""
        if not docs:
            return
        texts = [doc.to_text() for doc in docs]
        metadatas = [doc.to_chroma_metadata() for doc in docs]
        ids = [f"{doc.db_id}_{doc.table_name}" for doc in docs]
        raw_embeddings = self.embeddings.embed_documents(texts)
        embeddings = raw_embeddings[: len(ids)]
        self.collection.upsert(
            ids=ids,
            embeddings=cast(Any, embeddings),
            documents=texts,
            metadatas=cast(Any, metadatas),
        )

    def retrieve(
        self,
        question: str,
        db_id: str,
        top_k: int,
        similarity_threshold: float | None = None,
    ) -> list[SchemaDocument]:
        """Retrieve top-k schema documents relevant to the question, filtered by db_id.

        Args:
            similarity_threshold: If set, discard results whose cosine distance exceeds
                this value (0.0 = identical, 2.0 = opposite). Lower = stricter.
                Typical useful range: 0.3–0.8. Calibrate on your dev set.
                When None, all top_k results are returned regardless of distance.
        """
        query_embedding = self.embeddings.embed_query(question)
        results = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
            where={"db_id": db_id},
            include=["documents", "metadatas", "distances"],
        )
        docs: list[SchemaDocument] = []
        metadatas = results["metadatas"]
        documents = results["documents"]
        distances = results.get("distances")
        if metadatas is None or documents is None:
            return docs
        for i, (meta, text) in enumerate(zip(metadatas[0], documents[0], strict=False)):
            if similarity_threshold is not None and distances is not None:
                if distances[0][i] > similarity_threshold:
                    continue
            docs.append(
                _parse_schema_text(
                    text=text,
                    db_id=str(meta["db_id"]),
                    table_name=str(meta["table_name"]),
                )
            )
        return docs
