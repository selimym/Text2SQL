from typing import Protocol, runtime_checkable

from app.api.models import QueryRequest, QueryResponse


@runtime_checkable
class Pipeline(Protocol):
    def run(self, request: QueryRequest) -> QueryResponse: ...
