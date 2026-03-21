from fastapi import APIRouter

from app.api.models import QueryRequest, QueryResponse
from app.pipeline.baseline import BaselinePipeline

router = APIRouter()

# Module-level pipeline instance, set during app lifespan
_pipeline: BaselinePipeline | None = None


def set_pipeline(pipeline: BaselinePipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> BaselinePipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return _pipeline


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    return pipeline.run(request)
