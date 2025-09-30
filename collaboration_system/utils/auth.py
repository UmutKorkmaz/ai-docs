#!/usr/bin/env python3
"""
Authentication and authorization utilities

This module provides authentication functions, JWT handling,
permission management, and user session management.
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from functools import wraps
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .config import get_config
from ..models.collaboration_models import User, UserPermission

# Configuration
config = get_config()
security = HTTPBearer()

class AuthError(Exception):
    """Custom authentication error"""
    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    try:
        salt, password_hash = hashed_password.split(':')
        computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return computed_hash == password_hash
    except ValueError:
        return False

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.security.access_token_expire_minutes)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, config.security.secret_key, algorithm=config.security.algorithm)
    return encoded_jwt

def create_refresh_token(data: Dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=config.security.refresh_token_expire_days)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, config.security.secret_key, algorithm=config.security.algorithm)
    return encoded_jwt

def verify_token(token: str, token_type: str = "access") -> Dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, config.security.secret_key, algorithms=[config.security.algorithm])

        if payload.get("type") != token_type:
            raise AuthError("Invalid token type", 401)

        if datetime.utcnow() > datetime.fromtimestamp(payload["exp"]):
            raise AuthError("Token has expired", 401)

        return payload

    except jwt.PyJWTError as e:
        raise AuthError(f"Invalid token: {str(e)}", 401)

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current user ID from JWT token"""
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise AuthError("Invalid token payload", 401)
        return user_id
    except AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

def get_current_user(
    user_id: str = Depends(get_current_user_id),
    db: Session = None
) -> User:
    """Get current user from database"""
    if not db:
        raise HTTPException(status_code=500, detail="Database session not available")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="User account is inactive")

    return user

def require_permission(permission: UserPermission):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user (this would be handled by FastAPI dependency injection)
            user = kwargs.get('current_user')
            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Check if user has required permission
            # In a real implementation, you'd check user roles/permissions
            if not has_permission(user, permission):
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def has_permission(user: User, permission: UserPermission) -> bool:
    """Check if user has specific permission"""
    # Simplified permission check - in production, use role-based access control
    if permission == UserPermission.VIEWER:
        return True
    elif permission == UserPermission.COMMENTER:
        return user.permission in [UserPermission.COMMENTER, UserPermission.EDITOR, UserPermission.MODERATOR, UserPermission.ADMIN]
    elif permission == UserPermission.EDITOR:
        return user.permission in [UserPermission.EDITOR, UserPermission.MODERATOR, UserPermission.ADMIN]
    elif permission == UserPermission.MODERATOR:
        return user.permission in [UserPermission.MODERATOR, UserPermission.ADMIN]
    elif permission == UserPermission.ADMIN:
        return user.permission == UserPermission.ADMIN

    return False

async def verify_websocket_token(token: str) -> Dict:
    """Verify WebSocket token for real-time connections"""
    try:
        payload = verify_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise AuthError("Invalid token payload", 401)

        # In a real implementation, you'd fetch user info from database
        return {
            'user_id': user_id,
            'username': payload.get('username', 'Unknown'),
            'email': payload.get('email', ''),
            'permissions': payload.get('permissions', ['read', 'edit'])
        }
    except AuthError as e:
        raise e

def create_session_token(user: User, session_data: Dict = None) -> Dict:
    """Create session tokens for a user"""
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "permissions": [user.permission.value] if hasattr(user, 'permission') else ['read', 'edit']
        }
    )

    refresh_token = create_refresh_token(
        data={"sub": str(user.id)}
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": config.security.access_token_expire_minutes * 60,
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "avatar_url": user.avatar_url,
            "reputation_score": user.reputation_score,
            "is_active": user.is_active
        }
    }

def refresh_access_token(refresh_token: str) -> Dict:
    """Refresh access token using refresh token"""
    try:
        payload = verify_token(refresh_token, "refresh")
        user_id = payload.get("sub")

        # Create new access token
        new_access_token = create_access_token(
            data={"sub": user_id},
            expires_delta=timedelta(minutes=config.security.access_token_expire_minutes)
        )

        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": config.security.access_token_expire_minutes * 60
        }

    except AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

def validate_password_strength(password: str) -> List[str]:
    """Validate password strength and return list of issues"""
    issues = []

    if len(password) < config.security.password_min_length:
        issues.append(f"Password must be at least {config.security.password_min_length} characters long")

    if not any(c.isupper() for c in password):
        issues.append("Password must contain at least one uppercase letter")

    if not any(c.islower() for c in password):
        issues.append("Password must contain at least one lowercase letter")

    if not any(c.isdigit() for c in password):
        issues.append("Password must contain at least one digit")

    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        issues.append("Password must contain at least one special character")

    return issues

def generate_api_key() -> str:
    """Generate API key for programmatic access"""
    return f"aidocs_{secrets.token_urlsafe(32)}"

def validate_api_key(api_key: str, db: Session) -> Optional[User]:
    """Validate API key and return associated user"""
    # In a real implementation, you'd store and validate API keys in the database
    # This is a simplified version
    if not api_key.startswith('aidocs_'):
        return None

    # Extract user_id from API key (simplified)
    try:
        key_data = api_key[6:]  # Remove 'aidocs_' prefix
        user_id = key_data[:36]  # UUID is 36 characters
        return db.query(User).filter(User.id == user_id, User.is_active == True).first()
    except:
        return None

class RateLimiter:
    """Simple rate limiter for API endpoints"""

    def __init__(self, requests_per_window: int, window_seconds: int):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.requests = {}  # user_id -> list of timestamps

    def is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to make request"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)

        if user_id not in self.requests:
            self.requests[user_id] = []
            return True

        # Remove old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if req_time > window_start
        ]

        # Check if user has exceeded limit
        if len(self.requests[user_id]) >= self.requests_per_window:
            return False

        # Add current request
        self.requests[user_id].append(now)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_window=config.api.rate_limit_requests,
    window_seconds=config.api.rate_limit_window
)

def require_rate_limit(func):
    """Decorator to apply rate limiting"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get user ID from request context
        user_id = kwargs.get('user_id') or 'anonymous'

        if not rate_limiter.is_allowed(user_id):
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )

        return await func(*args, **kwargs)
    return wrapper

class SessionManager:
    """Manage user sessions and activity"""

    def __init__(self):
        self.active_sessions = {}  # session_id -> session_data
        self.user_sessions = {}   # user_id -> session_ids

    def create_session(self, user_id: str, session_data: Dict = None) -> str:
        """Create new session for user"""
        session_id = secrets.token_urlsafe(32)
        session_data = session_data or {}
        session_data.update({
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'is_active': True
        })

        self.active_sessions[session_id] = session_data

        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        return self.active_sessions.get(session_id)

    def update_session_activity(self, session_id: str):
        """Update session last activity"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['last_activity'] = datetime.utcnow()

    def end_session(self, session_id: str):
        """End a session"""
        if session_id in self.active_sessions:
            user_id = self.active_sessions[session_id]['user_id']
            del self.active_sessions[session_id]

            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]

    def cleanup_expired_sessions(self, timeout_hours: int = 24):
        """Clean up expired sessions"""
        cutoff_time = datetime.utcnow() - timedelta(hours=timeout_hours)

        expired_sessions = [
            session_id for session_id, data in self.active_sessions.items()
            if data['last_activity'] < cutoff_time
        ]

        for session_id in expired_sessions:
            self.end_session(session_id)

    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user"""
        return list(self.user_sessions.get(user_id, set()))

    def is_session_active(self, session_id: str) -> bool:
        """Check if session is active"""
        session = self.active_sessions.get(session_id)
        return session and session.get('is_active', False)

# Global session manager
session_manager = SessionManager()