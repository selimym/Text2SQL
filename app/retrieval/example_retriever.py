import asyncio
from typing import Any, cast

from chromadb.api import ClientAPI
from langchain_core.embeddings import Embeddings

from app.retrieval.example_doc import ExampleDocument


class ExampleRetriever:
    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str,
        chroma_client: ClientAPI,
    ) -> None:
        self.embeddings = embeddings
        self.collection = chroma_client.get_or_create_collection(collection_name)

    def _index_sync(self, docs: list[ExampleDocument]) -> None:
        if not docs:
            return
        texts = [doc.to_text() for doc in docs]
        metadatas = [doc.to_chroma_metadata() for doc in docs]
        ids = [f"{doc.db_id}_{i}_{doc.question[:30]}" for i, doc in enumerate(docs)]
        raw_embeddings = self.embeddings.embed_documents(texts)
        embeddings = raw_embeddings[: len(ids)]
        self.collection.upsert(
            ids=ids,
            embeddings=cast(Any, embeddings),
            documents=texts,
            metadatas=cast(Any, metadatas),
        )

    async def index(self, docs: list[ExampleDocument]) -> None:
        """Index example documents into ChromaDB."""
        await asyncio.to_thread(self._index_sync, docs)

    def _retrieve_sync(self, question: str, db_id: str | None, top_k: int) -> list[ExampleDocument]:
        query_embedding = self.embeddings.embed_query(question)
        where = {"db_id": db_id} if db_id is not None else None
        results = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=top_k,
            where=cast(Any, where),
        )
        docs: list[ExampleDocument] = []
        if results["metadatas"] is None:
            return docs
        for meta in results["metadatas"][0]:
            docs.append(
                ExampleDocument(
                    db_id=str(meta["db_id"]),
                    question=str(meta["question"]),
                    sql=str(meta["sql"]),
                )
            )
        return docs

    async def retrieve(self, question: str, db_id: str | None, top_k: int) -> list[ExampleDocument]:
        """Retrieve top-k example documents relevant to the question."""
        return await asyncio.to_thread(self._retrieve_sync, question, db_id, top_k)
