"""API key authentication middleware for the extraction service."""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import settings

security = HTTPBearer()


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Verify the Bearer token matches the configured SERVICE_API_KEY.

    Returns the API key on success, raises 401 on failure.
    """
    if credentials.credentials != settings.SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials
