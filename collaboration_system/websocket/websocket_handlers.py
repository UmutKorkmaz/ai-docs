#!/usr/bin/env python3
"""
WebSocket handlers for real-time collaboration

This module handles WebSocket connections and message routing for real-time features
including document editing, chat, user presence, and collaborative interactions.
"""

import json
import asyncio
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of WebSocket messages"""
    # Document collaboration
    EDIT = "edit"
    CURSOR_UPDATE = "cursor_update"
    SELECTION_UPDATE = "selection_update"
    DOCUMENT_STATE = "document_state"
    VERSION_UPDATE = "version_update"

    # User presence
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    USER_PRESENCE = "user_presence"
    TYPING_INDICATOR = "typing_indicator"

    # Chat and comments
    CHAT_MESSAGE = "chat_message"
    COMMENT_ADDED = "comment_added"
    COMMENT_UPDATED = "comment_updated"
    COMMENT_DELETED = "comment_deleted"

    # Q&A and discussions
    QUESTION_POSTED = "question_posted"
    ANSWER_POSTED = "answer_posted"
    VOTE_CAST = "vote_cast"

    # Study groups
    GROUP_MESSAGE = "group_message"
    GROUP_ACTIVITY = "group_activity"
    SESSION_INVITE = "session_invite"

    # System notifications
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class ConnectedUser:
    """Information about a connected WebSocket user"""
    websocket: WebSocket
    user_id: str
    username: str
    session_id: str
    document_id: str
    cursor_position: Optional[Dict] = None
    is_typing: bool = False
    last_activity: datetime = None
    permissions: List[str] = None

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()
        if self.permissions is None:
            self.permissions = ["read", "edit"]

class WebSocketManager:
    """
    Manages WebSocket connections and real-time messaging
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.active_connections: Dict[str, List[ConnectedUser]] = {}  # document_id -> users
        self.user_connections: Dict[str, ConnectedUser] = {}  # user_id -> connection
        self.session_connections: Dict[str, List[ConnectedUser]] = {}  # session_id -> users

        # Message handlers
        self.message_handlers = {
            MessageType.EDIT: self._handle_edit,
            MessageType.CURSOR_UPDATE: self._handle_cursor_update,
            MessageType.SELECTION_UPDATE: self._handle_selection_update,
            MessageType.CHAT_MESSAGE: self._handle_chat_message,
            MessageType.TYPING_INDICATOR: self._handle_typing_indicator,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.DOCUMENT_REQUEST: self._handle_document_request,
            MessageType.COMMENT_ADDED: self._handle_comment_added,
            MessageType.QUESTION_POSTED: self._handle_question_posted,
            MessageType.ANSWER_POSTED: self._handle_answer_posted,
            MessageType.VOTE_CAST: self._handle_vote_cast,
        }

    async def connect(self, websocket: WebSocket, session_id: str, user_info: Dict):
        """Accept and initialize WebSocket connection"""
        await websocket.accept()

        # Create connected user object
        connected_user = ConnectedUser(
            websocket=websocket,
            user_id=user_info['user_id'],
            username=user_info['username'],
            session_id=session_id,
            document_id=user_info.get('document_id', ''),
            cursor_position=None,
            permissions=user_info.get('permissions', ['read', 'edit'])
        )

        # Store connections
        if connected_user.document_id not in self.active_connections:
            self.active_connections[connected_user.document_id] = []
        self.active_connections[connected_user.document_id].append(connected_user)

        self.user_connections[connected_user.user_id] = connected_user

        if session_id not in self.session_connections:
            self.session_connections[session_id] = []
        self.session_connections[session_id].append(connected_user)

        # Notify other users
        await self._broadcast_to_document(connected_user.document_id, {
            'type': MessageType.USER_JOINED.value,
            'user': {
                'user_id': connected_user.user_id,
                'username': connected_user.username,
                'permissions': connected_user.permissions
            },
            'timestamp': datetime.utcnow().isoformat()
        }, exclude_user=connected_user.user_id)

        logger.info(f"User {connected_user.user_id} connected to session {session_id}")

        # Send current document state
        await self._send_document_state(connected_user)

    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        # Find and remove the connection
        disconnected_user = None
        for doc_id, users in self.active_connections.items():
            for user in users:
                if user.websocket == websocket:
                    disconnected_user = user
                    users.remove(user)
                    break

        if disconnected_user:
            # Remove from other mappings
            if disconnected_user.user_id in self.user_connections:
                del self.user_connections[disconnected_user.user_id]

            if disconnected_user.session_id in self.session_connections:
                session_users = self.session_connections[disconnected_user.session_id]
                if disconnected_user in session_users:
                    session_users.remove(disconnected_user)

            # Notify other users
            await self._broadcast_to_document(disconnected_user.document_id, {
                'type': MessageType.USER_LEFT.value,
                'user_id': disconnected_user.user_id,
                'timestamp': datetime.utcnow().isoformat()
            })

            logger.info(f"User {disconnected_user.user_id} disconnected")

    async def handle_message(self, websocket: WebSocket, message: Dict):
        """Handle incoming WebSocket message"""
        try:
            message_type = message.get('type')
            if not message_type:
                await self._send_error(websocket, "Message type is required")
                return

            # Find the user
            user = self._get_user_by_websocket(websocket)
            if not user:
                await self._send_error(websocket, "User not found")
                return

            # Update last activity
            user.last_activity = datetime.utcnow()

            # Route to appropriate handler
            handler = self.message_handlers.get(MessageType(message_type))
            if handler:
                await handler(user, message)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await self._send_error(websocket, f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(websocket, "Internal server error")

    async def _handle_edit(self, user: ConnectedUser, message: Dict):
        """Handle document edit operation"""
        edit_data = message.get('edit', {})
        operation = edit_data.get('operation')
        position = edit_data.get('position')
        text = edit_data.get('text', '')

        if not all([operation, position is not None]):
            await self._send_error(user.websocket, "Edit operation, position, and text are required")
            return

        # Validate user permissions
        if 'edit' not in user.permissions:
            await self._send_error(user.websocket, "No edit permission")
            return

        # Broadcast to other users in the document
        edit_message = {
            'type': MessageType.EDIT.value,
            'edit': {
                'operation': operation,
                'position': position,
                'text': text,
                'user_id': user.user_id,
                'username': user.username,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self._broadcast_to_document(user.document_id, edit_message, exclude_user=user.user_id)

        # Store edit in Redis for persistence
        await self._store_edit(user.document_id, edit_data)

    async def _handle_cursor_update(self, user: ConnectedUser, message: Dict):
        """Handle cursor position update"""
        cursor_data = message.get('cursor', {})
        user.cursor_position = cursor_data

        # Broadcast to other users
        await self._broadcast_to_document(user.document_id, {
            'type': MessageType.CURSOR_UPDATE.value,
            'user_id': user.user_id,
            'username': user.username,
            'cursor': cursor_data,
            'timestamp': datetime.utcnow().isoformat()
        }, exclude_user=user.user_id)

    async def _handle_selection_update(self, user: ConnectedUser, message: Dict):
        """Handle text selection update"""
        selection_data = message.get('selection', {})

        # Broadcast to other users
        await self._broadcast_to_document(user.document_id, {
            'type': MessageType.SELECTION_UPDATE.value,
            'user_id': user.user_id,
            'username': user.username,
            'selection': selection_data,
            'timestamp': datetime.utcnow().isoformat()
        }, exclude_user=user.user_id)

    async def _handle_chat_message(self, user: ConnectedUser, message: Dict):
        """Handle chat message"""
        chat_data = message.get('chat', {})
        message_text = chat_data.get('message', '')
        message_type = chat_data.get('message_type', 'text')

        if not message_text.strip():
            await self._send_error(user.websocket, "Message text is required")
            return

        # Create chat message
        chat_message = {
            'type': MessageType.CHAT_MESSAGE.value,
            'chat': {
                'id': str(uuid.uuid4()),
                'user_id': user.user_id,
                'username': user.username,
                'message': message_text,
                'message_type': message_type,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        # Broadcast to all users in the session
        await self._broadcast_to_session(user.session_id, chat_message)

        # Store in Redis
        await self._store_chat_message(user.session_id, chat_message['chat'])

    async def _handle_typing_indicator(self, user: ConnectedUser, message: Dict):
        """Handle typing indicator"""
        is_typing = message.get('is_typing', False)
        user.is_typing = is_typing

        # Broadcast to other users
        await self._broadcast_to_document(user.document_id, {
            'type': MessageType.TYPING_INDICATOR.value,
            'user_id': user.user_id,
            'username': user.username,
            'is_typing': is_typing,
            'timestamp': datetime.utcnow().isoformat()
        }, exclude_user=user.user_id)

    async def _handle_heartbeat(self, user: ConnectedUser, message: Dict):
        """Handle heartbeat/ping"""
        user.last_activity = datetime.utcnow()
        await user.websocket.send_json({
            'type': MessageType.HEARTBEAT.value,
            'timestamp': datetime.utcnow().isoformat()
        })

    async def _handle_document_request(self, user: ConnectedUser, message: Dict):
        """Handle document state request"""
        await self._send_document_state(user)

    async def _handle_comment_added(self, user: ConnectedUser, message: Dict):
        """Handle comment addition"""
        comment_data = message.get('comment', {})

        # Broadcast to document participants
        await self._broadcast_to_document(user.document_id, {
            'type': MessageType.COMMENT_ADDED.value,
            'comment': {
                **comment_data,
                'user_id': user.user_id,
                'username': user.username,
                'timestamp': datetime.utcnow().isoformat()
            }
        })

    async def _handle_question_posted(self, user: ConnectedUser, message: Dict):
        """Handle Q&A question posting"""
        question_data = message.get('question', {})

        # Broadcast to relevant users (e.g., topic experts)
        await self._broadcast_to_topic_experts(question_data.get('topic', ''), {
            'type': MessageType.QUESTION_POSTED.value,
            'question': {
                **question_data,
                'user_id': user.user_id,
                'username': user.username,
                'timestamp': datetime.utcnow().isoformat()
            }
        })

    async def _handle_answer_posted(self, user: ConnectedUser, message: Dict):
        """Handle Q&A answer posting"""
        answer_data = message.get('answer', {})
        question_id = answer_data.get('question_id')

        # Broadcast to question participants
        await self._broadcast_to_question_participants(question_id, {
            'type': MessageType.ANSWER_POSTED.value,
            'answer': {
                **answer_data,
                'user_id': user.user_id,
                'username': user.username,
                'timestamp': datetime.utcnow().isoformat()
            }
        })

    async def _handle_vote_cast(self, user: ConnectedUser, message: Dict):
        """Handle vote casting"""
        vote_data = message.get('vote', {})
        target_type = vote_data.get('target_type')  # comment, answer, question
        target_id = vote_data.get('target_id')
        vote_type = vote_data.get('vote_type')  # up, down

        # Notify the target user
        await self._notify_vote_target(target_type, target_id, {
            'type': MessageType.VOTE_CAST.value,
            'vote': {
                'voter_id': user.user_id,
                'voter_username': user.username,
                'vote_type': vote_type,
                'timestamp': datetime.utcnow().isoformat()
            }
        })

    async def _send_document_state(self, user: ConnectedUser):
        """Send current document state to user"""
        try:
            # Get document from Redis or database
            document_content = await self._get_document_content(user.document_id)

            await user.websocket.send_json({
                'type': MessageType.DOCUMENT_STATE.value,
                'document': {
                    'id': user.document_id,
                    'content': document_content,
                    'participants': [
                        {
                            'user_id': u.user_id,
                            'username': u.username,
                            'is_typing': u.is_typing,
                            'cursor_position': u.cursor_position
                        }
                        for u in self.active_connections.get(user.document_id, [])
                        if u.user_id != user.user_id
                    ]
                },
                'timestamp': datetime.utcnow().isoformat()
            })

        except Exception as e:
            logger.error(f"Error sending document state: {e}")
            await self._send_error(user.websocket, "Failed to load document state")

    async def _broadcast_to_document(self, document_id: str, message: Dict, exclude_user: str = None):
        """Broadcast message to all users in a document"""
        if document_id not in self.active_connections:
            return

        for user in self.active_connections[document_id]:
            if exclude_user and user.user_id == exclude_user:
                continue

            try:
                await user.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user.user_id}: {e}")

    async def _broadcast_to_session(self, session_id: str, message: Dict, exclude_user: str = None):
        """Broadcast message to all users in a session"""
        if session_id not in self.session_connections:
            return

        for user in self.session_connections[session_id]:
            if exclude_user and user.user_id == exclude_user:
                continue

            try:
                await user.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user.user_id}: {e}")

    async def _broadcast_to_topic_experts(self, topic: str, message: Dict):
        """Broadcast message to topic experts"""
        # Find users with expertise in the topic
        for user_id, user in self.user_connections.items():
            if 'expert' in user.permissions:  # Simplified expertise check
                try:
                    await user.websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to expert {user_id}: {e}")

    async def _broadcast_to_question_participants(self, question_id: str, message: Dict):
        """Broadcast message to question participants"""
        # In a real implementation, this would query the database for participants
        # For now, broadcast to all connected users
        for user in self.user_connections.values():
            try:
                await user.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user.user_id}: {e}")

    async def _notify_vote_target(self, target_type: str, target_id: str, message: Dict):
        """Notify the target of a vote"""
        # In a real implementation, this would look up the target user
        # For now, this is a placeholder
        pass

    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to client"""
        await websocket.send_json({
            'type': MessageType.ERROR.value,
            'error': error_message,
            'timestamp': datetime.utcnow().isoformat()
        })

    def _get_user_by_websocket(self, websocket: WebSocket) -> Optional[ConnectedUser]:
        """Get user by WebSocket connection"""
        for user in self.user_connections.values():
            if user.websocket == websocket:
                return user
        return None

    async def _store_edit(self, document_id: str, edit_data: Dict):
        """Store edit in Redis"""
        try:
            await self.redis_client.lpush(
                f"edits:{document_id}",
                json.dumps({
                    **edit_data,
                    'timestamp': datetime.utcnow().isoformat()
                })
            )
            # Keep only last 1000 edits
            await self.redis_client.ltrim(f"edits:{document_id}", 0, 999)
        except Exception as e:
            logger.error(f"Error storing edit: {e}")

    async def _store_chat_message(self, session_id: str, message: Dict):
        """Store chat message in Redis"""
        try:
            await self.redis_client.lpush(
                f"chat:{session_id}",
                json.dumps(message)
            )
            # Keep only last 500 messages
            await self.redis_client.ltrim(f"chat:{session_id}", 0, 499)
        except Exception as e:
            logger.error(f"Error storing chat message: {e}")

    async def _get_document_content(self, document_id: str) -> str:
        """Get document content from Redis or database"""
        try:
            content = await self.redis_client.get(f"document:{document_id}")
            if content:
                return content.decode('utf-8')
            # In a real implementation, fall back to database
            return ""
        except Exception as e:
            logger.error(f"Error getting document content: {e}")
            return ""

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'active_documents': len(self.active_connections),
            'active_sessions': len(self.session_connections),
            'total_users': len(self.user_connections),
            'connections_per_document': {
                doc_id: len(users) for doc_id, users in self.active_connections.items()
            }
        }

    async def cleanup_inactive_connections(self, timeout_minutes: int = 30):
        """Clean up inactive connections"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)

        for user_id, user in list(self.user_connections.items()):
            if user.last_activity < cutoff_time:
                await self.disconnect(user.websocket)

# Global WebSocket manager
websocket_manager = WebSocketManager()