"""JWT authentication for API endpoints."""

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

# JWT config
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
ENV_USERNAME = "AUTH_USERNAME"
ENV_PASSWORD_HASH = "AUTH_PASSWORD_HASH"
ENV_JWT_SECRET = "JWT_SECRET"

# Default dev credentials (bcrypt hash of "admin")
# In production, set AUTH_USERNAME, AUTH_PASSWORD_HASH, JWT_SECRET
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD_HASH = "$2b$12$CKJfD5WFmBI8gLjXubZT4ONIq9DbaPGaisdF9ueOQfGTg7fMhxoIm"  # "admin"

security = HTTPBearer(auto_error=False)


def _get_jwt_secret() -> str:
    secret = os.getenv(ENV_JWT_SECRET)
    if not secret:
        secret = "dev-secret-change-in-production"
    return secret


def _get_expected_username() -> str:
    return os.getenv(ENV_USERNAME, DEFAULT_USERNAME)


def _get_expected_password_hash() -> str:
    return os.getenv(ENV_PASSWORD_HASH, DEFAULT_PASSWORD_HASH)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify plain password against bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def hash_password(password: str) -> str:
    """Hash password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, _get_jwt_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    """Decode and validate JWT. Returns payload or None on failure."""
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def authenticate_user(username: str, password: str) -> bool:
    """Verify username and password against configured credentials."""
    expected_user = _get_expected_username()
    expected_hash = _get_expected_password_hash()
    return username == expected_user and verify_password(password, expected_hash)


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(security),
    ],
) -> dict:
    """FastAPI dependency: require valid Bearer token, return user payload."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload
