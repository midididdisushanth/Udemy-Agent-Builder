"""
tests/test_documents.py — Tests for document ingestion endpoints.
Run with: pytest tests/test_documents.py -v
"""
import os
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_CONTENT = """
DocIntel is a document intelligence platform that uses RAG to answer questions.
It combines ChromaDB for vector storage with OpenAI GPT-4o for answer generation.
The system supports text and file ingestion, semantic search, and synthetic QA generation.
""".strip()

SAMPLE_FILENAME = "test_ingest.txt"


class TestDocumentIngestion:

    def test_ingest_text_success(self):
        """Ingest a valid text document."""
        response = client.post("/documents/ingest/text", json={
            "content": SAMPLE_CONTENT,
            "filename": SAMPLE_FILENAME,
        })
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == SAMPLE_FILENAME
        assert data["total_chunks"] >= 1
        assert data["total_chars"] > 0

    def test_ingest_text_too_short(self):
        """Reject content that is too short."""
        response = client.post("/documents/ingest/text", json={
            "content": "Hi",
            "filename": "short.txt",
        })
        assert response.status_code == 422

    def test_ingest_duplicate_returns_existing(self):
        """Re-ingesting the same content returns the existing document."""
        # First ingest
        client.post("/documents/ingest/text", json={
            "content": SAMPLE_CONTENT,
            "filename": SAMPLE_FILENAME,
        })
        # Second ingest — same content
        response = client.post("/documents/ingest/text", json={
            "content": SAMPLE_CONTENT,
            "filename": SAMPLE_FILENAME,
        })
        assert response.status_code == 200
        data = response.json()
        assert "duplicate" in data["message"].lower()

    def test_ingest_file_txt(self):
        """Upload a .txt file for ingestion."""
        test_file_path = os.path.join(os.path.dirname(__file__), "test_document.txt")
        with open(test_file_path, "rb") as f:
            response = client.post(
                "/documents/ingest/file",
                files={"file": ("test_document.txt", f, "text/plain")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["total_chunks"] >= 1

    def test_ingest_file_non_txt_rejected(self):
        """Reject non-.txt file uploads."""
        response = client.post(
            "/documents/ingest/file",
            files={"file": ("doc.pdf", b"fake pdf content", "application/pdf")},
        )
        assert response.status_code == 400

    def test_list_documents(self):
        """List endpoint returns a list."""
        response = client.get("/documents/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_document_not_found(self):
        """404 for unknown document ID."""
        response = client.get("/documents/nonexistent-id-00000")
        assert response.status_code == 404

    def test_delete_document_not_found(self):
        """404 for deleting unknown document ID."""
        response = client.delete("/documents/nonexistent-id-00000")
        assert response.status_code == 404
