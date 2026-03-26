import asyncio
from typing import Any, cast

from chromadb.api import ClientAPI
from langchain_core.embeddings import Embeddings

from app.db.schema_doc import SchemaDocument


class SchemaRetriever:
    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str,
        chroma_client: ClientAPI,
    ) -> None:
        self.embeddings = embeddings
        self.collection = chroma_client.get_or_create_collection(collection_name)

    def _index_sync(self, docs: list[SchemaDocument]) -> None:
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

    async def index(self, docs: list[SchemaDocument]) -> None:
        """Index schema documents into ChromaDB."""
        await asyncio.to_thread(self._index_sync, docs)

    def _retrieve_sync(self, question: str, db_id: str, top_k: int) -> list[SchemaDocument]:
        query_embedding = self.embeddings.embed_query(question)
        results = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
            where={"db_id": db_id},
        )
        docs: list[SchemaDocument] = []
        metadatas = results["metadatas"]
        documents = results["documents"]
        if metadatas is None or documents is None:
            return docs
        for meta, _text in zip(metadatas[0], documents[0], strict=False):
            docs.append(
                SchemaDocument(
                    db_id=str(meta["db_id"]),
                    table_name=str(meta["table_name"]),
                    columns=[],  # metadata doesn't store full columns; callers may re-introspect
                    foreign_keys=[],
                )
            )
        return docs

    async def retrieve(self, question: str, db_id: str, top_k: int) -> list[SchemaDocument]:
        """Retrieve top-k schema documents relevant to the question, filtered by db_id."""
        return await asyncio.to_thread(self._retrieve_sync, question, db_id, top_k)
