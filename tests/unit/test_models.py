import uuid

from app.api.models import ErrorResponse, QueryRequest, QueryResponse


def test_query_request_required_fields() -> None:
    req = QueryRequest(question="How many singers?", db_id="concert_singer")
    assert req.question == "How many singers?"
    assert req.db_id == "concert_singer"
    assert req.top_k_schema == 5
    assert req.top_k_examples == 3


def test_query_response_auto_trace_id() -> None:
    resp = QueryResponse(question="q", generated_sql="SELECT 1", answer="1")
    assert resp.trace_id is not None
    uuid.UUID(resp.trace_id)  # validates it's a valid UUID


def test_query_response_flags_default_empty() -> None:
    resp = QueryResponse(question="q", generated_sql="SELECT 1", answer="1")
    assert resp.flags == []


def test_error_response_auto_trace_id() -> None:
    err = ErrorResponse(error="something went wrong")
    assert err.trace_id is not None
    uuid.UUID(err.trace_id)


def test_query_response_new_fields_default_none() -> None:
    resp = QueryResponse(question="q", generated_sql="SELECT 1", answer="1")
    assert resp.step_timings is None
    assert resp.retry_count is None


def test_query_response_new_fields_accept_values() -> None:
    resp = QueryResponse(
        question="q",
        generated_sql="SELECT 1",
        answer="1",
        step_timings={"retrieve": 0.1, "generate": 0.5},
        retry_count=1,
    )
    assert resp.step_timings == {"retrieve": 0.1, "generate": 0.5}
    assert resp.retry_count == 1
