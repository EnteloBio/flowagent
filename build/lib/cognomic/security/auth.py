"""Authentication and authorization for Cognomic."""
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from ..config.settings import settings

# Security schemas
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER)


class Token(BaseModel):
    """Token model."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: list[str] = []


class SecurityService:
    """Security service for authentication and authorization."""

    def __init__(self) -> None:
        """Initialize security service."""
        self.secret_key = settings.SECRET_KEY.get_secret_value()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )
            
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, self.secret_key, algorithm=self.algorithm
        )
        return encoded_jwt

    def verify_token(self, token: str) -> TokenData:
        """Verify a token and return its data."""
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm]
            )
            username: str = payload.get("sub")
            scopes: list[str] = payload.get("scopes", [])
            
            if username is None:
                raise HTTPException(
                    status_code=401,
                    detail="Could not validate credentials",
                )
                
            return TokenData(username=username, scopes=scopes)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired",
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
            )

    async def get_current_user(
        self, token: str = Depends(api_key_header)
    ) -> TokenData:
        """Get current user from token."""
        return self.verify_token(token)

    def verify_scope(self, required_scope: str) -> None:
        """Verify if a user has the required scope."""
        async def scope_validator(
            current_user: TokenData = Security(get_current_user)
        ) -> None:
            if required_scope not in current_user.scopes:
                raise HTTPException(
                    status_code=403,
                    detail=f"Not enough permissions. Required scope: {required_scope}",
                )
        return scope_validator


# Global security service instance
security_service = SecurityService()
