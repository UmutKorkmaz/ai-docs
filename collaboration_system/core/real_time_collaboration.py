#!/usr/bin/env python3
"""
Real-time Collaboration System for AI Documentation Platform

This module provides the foundation for real-time collaboration features including:
- Multi-user simultaneous editing with operational transformation
- Real-time synchronization and conflict resolution
- WebSocket-based communication backbone
- Version control integration
- User presence and activity tracking
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import HTTPException
import redis
from redis.commands.json.path import Path
import jwt
from cryptography.fernet import Fernet
import hashlib
import hmac

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserPermission(Enum):
    """User permission levels for collaboration"""
    VIEWER = "viewer"
    COMMENTER = "commenter"
    EDITOR = "editor"
    MODERATOR = "moderator"
    ADMIN = "admin"

class EditOperation(Enum):
    """Types of edit operations"""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"
    ANNOTATE = "annotate"
    COMMENT = "comment"
    SUGGESTION = "suggestion"

@dataclass
class User:
    """User information for collaboration sessions"""
    user_id: str
    username: str
    email: str
    avatar_url: Optional[str] = None
    permission: UserPermission = UserPermission.EDITOR
    is_active: bool = True
    last_seen: datetime = None

    def __post_init__(self):
        if self.last_seen is None:
            self.last_seen = datetime.utcnow()

@dataclass
class DocumentPosition:
    """User cursor/selection position in document"""
    line: int
    column: int
    selection_start: int
    selection_end: int
    document_id: str

@dataclass
class EditChange:
    """Single edit operation with metadata"""
    operation: EditOperation
    position: int
    text: str
    user_id: str
    timestamp: datetime
    version: int
    document_id: str
    parent_change_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if not hasattr(self, 'change_id'):
            self.change_id = str(uuid.uuid4())

@dataclass
class CollaborationSession:
    """Active collaboration session for a document"""
    session_id: str
    document_id: str
    document_content: str
    participants: Dict[str, User]
    edit_history: List[EditChange]
    current_version: int
    created_at: datetime
    last_activity: datetime

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()

class OperationalTransformation:
    """
    Operational Transformation for conflict resolution in concurrent editing
    Implements basic OT operations for text collaboration
    """

    @staticmethod
    def transform(op1: EditChange, op2: EditChange) -> Tuple[EditChange, EditChange]:
        """Transform two concurrent operations to resolve conflicts"""
        # Simple implementation - can be extended with more sophisticated OT algorithms
        if op1.operation == EditOperation.INSERT and op2.operation == EditOperation.INSERT:
            return OperationalTransformation._transform_insert_insert(op1, op2)
        elif op1.operation == EditOperation.INSERT and op2.operation == EditOperation.DELETE:
            return OperationalTransformation._transform_insert_delete(op1, op2)
        elif op1.operation == EditOperation.DELETE and op2.operation == EditOperation.INSERT:
            return OperationalTransformation._transform_delete_insert(op1, op2)
        elif op1.operation == EditOperation.DELETE and op2.operation == EditOperation.DELETE:
            return OperationalTransformation._transform_delete_delete(op1, op2)
        else:
            return op1, op2

    @staticmethod
    def _transform_insert_insert(op1: EditChange, op2: EditChange) -> Tuple[EditChange, EditChange]:
        """Transform two insert operations"""
        if op1.position <= op2.position:
            # op1 comes before op2, no transformation needed
            return op1, EditChange(
                operation=op2.operation,
                position=op2.position + len(op1.text),
                text=op2.text,
                user_id=op2.user_id,
                timestamp=op2.timestamp,
                version=op2.version,
                document_id=op2.document_id
            )
        else:
            # op2 comes before op1
            return EditChange(
                operation=op1.operation,
                position=op1.position + len(op2.text),
                text=op1.text,
                user_id=op1.user_id,
                timestamp=op1.timestamp,
                version=op1.version,
                document_id=op1.document_id
            ), op2

    @staticmethod
    def _transform_insert_delete(op1: EditChange, op2: EditChange) -> Tuple[EditChange, EditChange]:
        """Transform insert and delete operations"""
        if op2.position <= op1.position:
            # Delete happens before insert
            return EditChange(
                operation=op1.operation,
                position=op1.position - min(op2.position, len(op2.text)),
                text=op1.text,
                user_id=op1.user_id,
                timestamp=op1.timestamp,
                version=op1.version,
                document_id=op1.document_id
            ), op2
        else:
            # Insert happens before delete
            return op1, op2

    @staticmethod
    def _transform_delete_insert(op1: EditChange, op2: EditChange) -> Tuple[EditChange, EditChange]:
        """Transform delete and insert operations"""
        if op1.position <= op2.position:
            # Delete happens before insert
            return op1, EditChange(
                operation=op2.operation,
                position=op2.position - min(op1.position, len(op1.text)),
                text=op2.text,
                user_id=op2.user_id,
                timestamp=op2.timestamp,
                version=op2.version,
                document_id=op2.document_id
            )
        else:
            # Insert happens before delete
            return op1, op2

    @staticmethod
    def _transform_delete_delete(op1: EditChange, op2: EditChange) -> Tuple[EditChange, EditChange]:
        """Transform two delete operations"""
        if op1.position <= op2.position:
            return op1, EditChange(
                operation=op2.operation,
                position=max(op2.position - len(op1.text), op1.position),
                text=op2.text,
                user_id=op2.user_id,
                timestamp=op2.timestamp,
                version=op2.version,
                document_id=op2.document_id
            )
        else:
            return EditChange(
                operation=op1.operation,
                position=max(op1.position - len(op2.text), op2.position),
                text=op1.text,
                user_id=op1.user_id,
                timestamp=op1.timestamp,
                version=op1.version,
                document_id=op1.document_id
            ), op2

class RealTimeCollaborationManager:
    """
    Main manager for real-time collaboration features
    Handles user sessions, document synchronization, and conflict resolution
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", secret_key: str = None):
        self.redis_client = redis.from_url(redis_url)
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        self.encryption_key = Fernet.generate_key() if secret_key is None else secret_key
        self.cipher_suite = Fernet(self.encryption_key)

    async def create_session(self, document_id: str, initial_content: str,
                           creator_id: str, creator_info: Dict) -> str:
        """Create a new collaboration session"""
        session_id = str(uuid.uuid4())

        # Create user object
        user = User(
            user_id=creator_id,
            username=creator_info.get('username', 'Unknown'),
            email=creator_info.get('email', ''),
            avatar_url=creator_info.get('avatar_url'),
            permission=UserPermission.ADMIN
        )

        # Create session
        session = CollaborationSession(
            session_id=session_id,
            document_id=document_id,
            document_content=initial_content,
            participants={creator_id: user},
            edit_history=[],
            current_version=1,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )

        # Store session
        self.active_sessions[session_id] = session
        self.websocket_connections[session_id] = []

        # Add user to session mapping
        if creator_id not in self.user_sessions:
            self.user_sessions[creator_id] = set()
        self.user_sessions[creator_id].add(session_id)

        # Store in Redis for persistence
        await self._store_session_in_redis(session)

        logger.info(f"Created collaboration session {session_id} for document {document_id}")
        return session_id

    async def join_session(self, session_id: str, user_id: str, user_info: Dict) -> bool:
        """Join an existing collaboration session"""
        if session_id not in self.active_sessions:
            # Try to load from Redis
            session = await self._load_session_from_redis(session_id)
            if session is None:
                return False
            self.active_sessions[session_id] = session

        session = self.active_sessions[session_id]

        # Create user object
        user = User(
            user_id=user_id,
            username=user_info.get('username', 'Unknown'),
            email=user_info.get('email', ''),
            avatar_url=user_info.get('avatar_url'),
            permission=UserPermission(user_info.get('permission', 'editor'))
        )

        # Add user to session
        session.participants[user_id] = user
        session.last_activity = datetime.utcnow()

        # Update user sessions mapping
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)

        # Notify other participants
        await self._broadcast_to_session(session_id, {
            'type': 'user_joined',
            'user': asdict(user),
            'timestamp': datetime.utcnow().isoformat()
        })

        logger.info(f"User {user_id} joined session {session_id}")
        return True

    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a collaboration session"""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        # Remove user from session
        if user_id in session.participants:
            del session.participants[user_id]
            session.last_activity = datetime.utcnow()

        # Remove from user sessions mapping
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)

        # Notify other participants
        await self._broadcast_to_session(session_id, {
            'type': 'user_left',
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat()
        })

        # Clean up empty sessions
        if len(session.participants) == 0:
            await self._cleanup_session(session_id)

        logger.info(f"User {user_id} left session {session_id}")
        return True

    async def apply_edit(self, session_id: str, edit_change: EditChange) -> bool:
        """Apply an edit change to the document"""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        # Validate user permission
        user = session.participants.get(edit_change.user_id)
        if not user or user.permission not in [UserPermission.EDITOR, UserPermission.MODERATOR, UserPermission.ADMIN]:
            return False

        # Apply operational transformation with concurrent edits
        # This is a simplified version - in production, you'd need more sophisticated OT
        transformed_edit = edit_change

        # Apply the edit to document content
        try:
            if edit_change.operation == EditOperation.INSERT:
                session.document_content = (
                    session.document_content[:edit_change.position] +
                    edit_change.text +
                    session.document_content[edit_change.position:]
                )
            elif edit_change.operation == EditOperation.DELETE:
                session.document_content = (
                    session.document_content[:edit_change.position] +
                    session.document_content[edit_change.position + len(edit_change.text):]
                )
            elif edit_change.operation == EditOperation.REPLACE:
                session.document_content = (
                    session.document_content[:edit_change.position] +
                    edit_change.text +
                    session.document_content[edit_change.position + len(edit_change.text):]
                )

            # Add to edit history
            edit_change.version = session.current_version
            session.edit_history.append(transformed_edit)
            session.current_version += 1
            session.last_activity = datetime.utcnow()

            # Broadcast to other participants
            await self._broadcast_to_session(session_id, {
                'type': 'edit_applied',
                'edit': asdict(transformed_edit),
                'version': session.current_version,
                'document_content': session.document_content,
                'timestamp': datetime.utcnow().isoformat()
            })

            logger.info(f"Applied edit by user {edit_change.user_id} in session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error applying edit: {e}")
            return False

    async def handle_websocket_connection(self, websocket: WebSocket, session_id: str, user_id: str):
        """Handle WebSocket connection for real-time updates"""
        await websocket.accept()

        # Add to connection list
        if session_id not in self.websocket_connections:
            self.websocket_connections[session_id] = []
        self.websocket_connections[session_id].append(websocket)

        try:
            # Send current document state
            session = self.active_sessions.get(session_id)
            if session:
                await websocket.send_json({
                    'type': 'document_state',
                    'content': session.document_content,
                    'version': session.current_version,
                    'participants': [asdict(u) for u in session.participants.values()],
                    'timestamp': datetime.utcnow().isoformat()
                })

            # Listen for messages
            while True:
                try:
                    data = await websocket.receive_json()
                    await self._handle_websocket_message(session_id, user_id, data)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    break

        finally:
            # Remove from connection list
            if session_id in self.websocket_connections:
                self.websocket_connections[session_id].remove(websocket)
                if not self.websocket_connections[session_id]:
                    del self.websocket_connections[session_id]

    async def _handle_websocket_message(self, session_id: str, user_id: str, data: Dict):
        """Handle incoming WebSocket messages"""
        message_type = data.get('type')

        if message_type == 'edit':
            edit_change = EditChange(
                operation=EditOperation(data['operation']),
                position=data['position'],
                text=data['text'],
                user_id=user_id,
                timestamp=datetime.utcnow(),
                version=0,  # Will be set by apply_edit
                document_id=session_id
            )
            await self.apply_edit(session_id, edit_change)

        elif message_type == 'cursor_position':
            # Update user cursor position
            await self._broadcast_to_session(session_id, {
                'type': 'cursor_update',
                'user_id': user_id,
                'position': data['position'],
                'timestamp': datetime.utcnow().isoformat()
            }, exclude_user=user_id)

        elif message_type == 'comment':
            # Handle comment addition
            comment_data = data.get('comment', {})
            await self._add_comment(session_id, user_id, comment_data)

    async def _add_comment(self, session_id: str, user_id: str, comment_data: Dict):
        """Add a comment to the document"""
        session = self.active_sessions.get(session_id)
        if not session:
            return

        comment = {
            'comment_id': str(uuid.uuid4()),
            'user_id': user_id,
            'text': comment_data.get('text', ''),
            'position': comment_data.get('position', 0),
            'timestamp': datetime.utcnow().isoformat(),
            'replies': []
        }

        # Add comment to session (in a real implementation, this would be stored separately)
        session.last_activity = datetime.utcnow()

        # Broadcast to all participants
        await self._broadcast_to_session(session_id, {
            'type': 'comment_added',
            'comment': comment,
            'timestamp': datetime.utcnow().isoformat()
        })

    async def _broadcast_to_session(self, session_id: str, message: Dict, exclude_user: str = None):
        """Broadcast message to all participants in a session"""
        if session_id not in self.websocket_connections:
            return

        for websocket in self.websocket_connections[session_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

    async def _store_session_in_redis(self, session: CollaborationSession):
        """Store session data in Redis for persistence"""
        try:
            session_data = {
                'session_id': session.session_id,
                'document_id': session.document_id,
                'document_content': session.document_content,
                'current_version': session.current_version,
                'created_at': session.created_at.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'participants': {uid: asdict(user) for uid, user in session.participants.items()},
                'edit_history': [asdict(edit) for edit in session.edit_history[-100:]]  # Keep last 100 edits
            }

            # Encrypt sensitive data
            encrypted_data = self.cipher_suite.encrypt(json.dumps(session_data).encode())
            self.redis_client.setex(
                f"session:{session.session_id}",
                3600,  # 1 hour TTL
                encrypted_data
            )

        except Exception as e:
            logger.error(f"Error storing session in Redis: {e}")

    async def _load_session_from_redis(self, session_id: str) -> Optional[CollaborationSession]:
        """Load session data from Redis"""
        try:
            encrypted_data = self.redis_client.get(f"session:{session_id}")
            if not encrypted_data:
                return None

            session_data = json.loads(self.cipher_suite.decrypt(encrypted_data).decode())

            # Reconstruct session object
            participants = {
                uid: User(**user_data)
                for uid, user_data in session_data['participants'].items()
            }

            edit_history = [
                EditChange(**edit_data)
                for edit_data in session_data['edit_history']
            ]

            session = CollaborationSession(
                session_id=session_data['session_id'],
                document_id=session_data['document_id'],
                document_content=session_data['document_content'],
                participants=participants,
                edit_history=edit_history,
                current_version=session_data['current_version'],
                created_at=datetime.fromisoformat(session_data['created_at']),
                last_activity=datetime.fromisoformat(session_data['last_activity'])
            )

            return session

        except Exception as e:
            logger.error(f"Error loading session from Redis: {e}")
            return None

    async def _cleanup_session(self, session_id: str):
        """Clean up empty session"""
        if session_id in self.active_sessions:
            # Store final state before cleanup
            await self._store_session_in_redis(self.active_sessions[session_id])
            del self.active_sessions[session_id]

        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]

        # Remove from Redis
        self.redis_client.delete(f"session:{session_id}")

        logger.info(f"Cleaned up session {session_id}")

    def get_active_sessions(self) -> List[Dict]:
        """Get list of active sessions"""
        return [
            {
                'session_id': session.session_id,
                'document_id': session.document_id,
                'participant_count': len(session.participants),
                'created_at': session.created_at.isoformat(),
                'last_activity': session.last_activity.isoformat()
            }
            for session in self.active_sessions.values()
        ]

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a specific session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        return {
            'session_id': session.session_id,
            'document_id': session.document_id,
            'participants': [asdict(user) for user in session.participants.values()],
            'current_version': session.current_version,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'edit_count': len(session.edit_history)
        }

# Global instance
collaboration_manager = RealTimeCollaborationManager()