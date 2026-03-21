from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.models import QueryResponse
from app.main import create_app


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.mark.asyncio
async def test_health_check(app: FastAPI) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_query_endpoint(app: FastAPI) -> None:
    mock_response = QueryResponse(
        question="How many singers?",
        generated_sql="SELECT COUNT(*) FROM singer",
        answer="[[3]]",
    )
    with patch("app.api.routes.get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_response
        mock_get_pipeline.return_value = mock_pipeline
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={"question": "How many singers?", "db_id": "concert_singer"},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["generated_sql"] == "SELECT COUNT(*) FROM singer"
    assert "trace_id" in data
