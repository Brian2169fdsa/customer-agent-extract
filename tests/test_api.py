"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)

# Valid auth header for tests
AUTH_HEADER = {"Authorization": f"Bearer {settings.SERVICE_API_KEY}"}


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_ok(self):
        """Health endpoint should return 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data


class TestAuthMiddleware:
    """Test API key authentication."""

    def test_missing_auth_returns_401_or_403(self):
        """Request without auth header should be rejected."""
        response = client.post(
            "/extract",
            json={"content": "test", "content_type": "transcript"},
        )
        # FastAPI's HTTPBearer returns 401 when no credentials provided
        assert response.status_code in (401, 403)

    def test_invalid_key_returns_401(self):
        """Request with wrong API key should be rejected."""
        response = client.post(
            "/extract",
            json={"content": "test", "content_type": "transcript"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401

    def test_health_no_auth_required(self):
        """Health endpoint should not require authentication."""
        response = client.get("/health")
        assert response.status_code == 200


class TestExtractEndpoint:
    """Test the /extract endpoint request validation."""

    def test_empty_content_returns_empty_facts(self):
        """Empty content should return success with empty facts list."""
        response = client.post(
            "/extract",
            json={"content": "", "content_type": "transcript"},
            headers=AUTH_HEADER,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["facts"] == []

    def test_invalid_content_type_returns_422(self):
        """Invalid content_type should return 422 validation error."""
        response = client.post(
            "/extract",
            json={"content": "test", "content_type": "invalid_type"},
            headers=AUTH_HEADER,
        )
        assert response.status_code == 422

    def test_valid_request_schema(self):
        """Valid request body should pass Pydantic validation."""
        from app.schemas.extraction import ExtractionRequest

        request = ExtractionRequest(
            content="Meeting transcript content here",
            content_type="transcript",
            interaction_id="notion-page-123",
            priority="high",
        )
        assert request.content_type == "transcript"
        assert request.priority == "high"

    def test_default_content_type(self):
        """Default content_type should be transcript."""
        from app.schemas.extraction import ExtractionRequest

        request = ExtractionRequest(content="Some content")
        assert request.content_type == "transcript"

    def test_default_priority(self):
        """Default priority should be normal."""
        from app.schemas.extraction import ExtractionRequest

        request = ExtractionRequest(content="Some content")
        assert request.priority == "normal"
