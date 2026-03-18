"""
tests/test_chat.py — Tests for RAG chat, synthetic data, feedback, and stats.
Run with: pytest tests/test_chat.py -v
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestHealthAndStats:

    def test_health_check(self):
        """Health endpoint returns ok status."""
        response = client.get("/stats/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data

    def test_stats_schema(self):
        """Stats endpoint returns correct schema."""
        response = client.get("/stats/")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "total_queries" in data
        assert "vector_store" in data
        assert isinstance(data["total_documents"], int)

    def test_root_endpoint(self):
        """Root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "docs" in data


class TestRAGChat:

    def test_chat_request_schema(self):
        """Chat endpoint accepts valid request and returns correct schema."""
        response = client.post("/chat/", json={
            "query": "What is DocIntel and what does it do?",
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "answer" in data
        assert "sources" in data
        assert "tokens_used" in data
        assert "latency_ms" in data
        assert "query_log_id" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_chat_blank_query_rejected(self):
        """Blank queries are rejected."""
        response = client.post("/chat/", json={"query": "   ", "top_k": 5})
        assert response.status_code == 422

    def test_chat_history(self):
        """Chat history endpoint returns a list."""
        response = client.get("/chat/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_chat_with_custom_system_prompt(self):
        """Custom system prompt is accepted."""
        response = client.post("/chat/", json={
            "query": "Summarize the document briefly.",
            "top_k": 3,
            "system_prompt": "You are a concise summarizer. Reply in one sentence only.",
        })
        assert response.status_code == 200


class TestFeedback:

    def test_feedback_invalid_rating(self):
        """Only ratings of 1 or -1 are accepted."""
        response = client.post("/chat/feedback", json={
            "query_log_id": "some-id",
            "rating": 5,
        })
        assert response.status_code == 422

    def test_feedback_unknown_log_id(self):
        """404 for unknown query_log_id."""
        response = client.post("/chat/feedback", json={
            "query_log_id": "nonexistent-log-id",
            "rating": 1,
        })
        assert response.status_code == 404

    def test_feedback_summary(self):
        """Feedback summary returns counts."""
        response = client.get("/chat/feedback/summary")
        assert response.status_code == 200
        data = response.json()
        assert "thumbs_up" in data
        assert "thumbs_down" in data


class TestSyntheticData:

    def test_synthetic_unknown_document(self):
        """404 for unknown document_id."""
        response = client.post("/chat/synthetic/generate", json={
            "document_id": "nonexistent-doc-id",
            "num_pairs": 3,
        })
        assert response.status_code == 404

    def test_list_synthetic_empty(self):
        """Returns empty list for unknown document."""
        response = client.get("/chat/synthetic/nonexistent-doc-id")
        assert response.status_code == 200
        assert response.json() == []
