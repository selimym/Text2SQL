from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI

from app.api.routes import router, set_pipeline
from app.core.config import get_settings
from app.core.llm import get_llm
from app.core.logging import configure_logging
from app.db.executor import SQLExecutor
from app.db.validator import SQLValidator
from app.llm.generator import SQLGenerator
from app.pipeline.assembler import PromptAssembler
from app.pipeline.factory import build_pipeline
from app.retrieval.embeddings import get_embeddings
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embeddings = get_embeddings(settings.embedding_provider, settings.embedding_model)

    schema_retriever = SchemaRetriever(
        embeddings=embeddings,
        collection_name="spider_schemas",
        chroma_client=chroma_client,
    )
    example_retriever = ExampleRetriever(
        embeddings=embeddings,
        collection_name="spider_examples",
        chroma_client=chroma_client,
    )
    pipeline = build_pipeline(
        schema_retriever=schema_retriever,
        example_retriever=example_retriever,
        assembler=PromptAssembler(),
        generator=SQLGenerator(llm=get_llm(settings.llm_provider, settings.llm_model)),
        validator=SQLValidator(),
        executor=SQLExecutor(
            max_rows=settings.max_result_rows,
            timeout_seconds=settings.query_timeout_seconds,
        ),
        spider_data_dir=settings.spider_data_dir,
        variant=settings.pipeline_variant,
    )
    set_pipeline(pipeline)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Text2SQL", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
