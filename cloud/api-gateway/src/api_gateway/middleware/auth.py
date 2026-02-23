"""OAuth2/OIDC authentication middleware."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from api_gateway.config import Settings
from api_gateway.deps import get_settings

security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Verify JWT token and return decoded claims.

    In development mode (no OAuth2 issuer configured), returns a mock user.
    """
    if not settings.oauth2_issuer:
        # Dev mode: return mock user
        return {"sub": "dev-user", "role": "admin"}

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )


async def require_role(role: str):
    """Create a dependency that requires a specific role."""

    async def _check(claims: dict = Depends(verify_token)):
        user_role = claims.get("role", "")
        if user_role != role and user_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required",
            )
        return claims

    return _check
