"""
tests/test_search.py — Tests for semantic search endpoints.
Run with: pytest tests/test_search.py -v
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestSemanticSearch:

    def test_search_returns_valid_schema(self):
        """Search response contains required fields."""
        response = client.post("/search/", json={
            "query": "What is retrieval-augmented generation?",
            "top_k": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_found" in data
        assert isinstance(data["results"], list)

    def test_search_query_too_short(self):
        """Queries under 3 chars are rejected."""
        response = client.post("/search/", json={"query": "hi", "top_k": 5})
        assert response.status_code == 422

    def test_search_with_top_k(self):
        """top_k parameter is respected."""
        response = client.post("/search/", json={
            "query": "document processing pipeline",
            "top_k": 2,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 2

    def test_search_result_fields(self):
        """Each result contains required fields."""
        response = client.post("/search/", json={
            "query": "vector database embedding search",
            "top_k": 5,
        })
        assert response.status_code == 200
        results = response.json()["results"]
        for r in results:
            assert "chunk_id" in r
            assert "document_id" in r
            assert "filename" in r
            assert "content" in r
            assert "similarity_score" in r
            assert "chunk_index" in r
            assert 0.0 <= r["similarity_score"] <= 1.0

    def test_search_history(self):
        """Search history endpoint returns a list."""
        response = client.get("/search/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_search_with_document_filter(self):
        """Document ID filter is accepted (may return 0 results for unknown ID)."""
        response = client.post("/search/", json={
            "query": "FastAPI framework",
            "top_k": 5,
            "document_id": "nonexistent-doc-id",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_found"] == 0

    def test_search_top_k_bounds(self):
        """top_k must be between 1 and 20."""
        response = client.post("/search/", json={"query": "test query here", "top_k": 25})
        assert response.status_code == 422

        response = client.post("/search/", json={"query": "test query here", "top_k": 0})
        assert response.status_code == 422
