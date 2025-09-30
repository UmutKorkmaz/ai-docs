#!/usr/bin/env python3
"""
WebSocket Server for Real-time Collaboration

This module provides the main WebSocket server that handles real-time
communication for the collaboration platform.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Set, Optional
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

from .websocket_handlers import WebSocketManager, MessageType
from ..utils.auth import verify_websocket_token
from ..utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

class CollaborationWebSocketServer:
    """Main WebSocket server for collaboration features"""

    def __init__(self):
        self.manager = WebSocketManager(config.REDIS_URL)
        self.active_connections: Set[WebSocketServerProtocol] = set()
        self.user_sessions: Dict[str, Dict] = {}  # user_id -> session_info
        self.document_sessions: Dict[str, Set[str]] = {}  # document_id -> user_ids
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_processed': 0,
            'errors_handled': 0
        }

    async def register_connection(self, websocket: WebSocketServerProtocol):
        """Register a new WebSocket connection"""
        self.active_connections.add(websocket)
        self.stats['active_connections'] += 1
        self.stats['total_connections'] += 1
        logger.info(f"New connection registered. Total active: {self.stats['active_connections']}")

    async def unregister_connection(self, websocket: WebSocketServerProtocol):
        """Unregister a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.stats['active_connections'] -= 1

            # Clean up user sessions
            user_id = None
            for uid, session_info in self.user_sessions.items():
                if session_info.get('websocket') == websocket:
                    user_id = uid
                    break

            if user_id:
                await self.cleanup_user_session(user_id)

            logger.info(f"Connection unregistered. Total active: {self.stats['active_connections']}")

    async def cleanup_user_session(self, user_id: str):
        """Clean up user session data"""
        if user_id in self.user_sessions:
            session_info = self.user_sessions[user_id]
            document_id = session_info.get('document_id')
            session_id = session_info.get('session_id')

            # Remove from document sessions
            if document_id and document_id in self.document_sessions:
                self.document_sessions[document_id].discard(user_id)
                if not self.document_sessions[document_id]:
                    del self.document_sessions[document_id]

            # Notify other users
            if document_id:
                await self.broadcast_to_document(document_id, {
                    'type': MessageType.USER_LEFT.value,
                    'user_id': user_id,
                    'timestamp': datetime.utcnow().isoformat()
                })

            # Remove from manager
            await self.manager.disconnect(session_info.get('websocket'))

            # Remove from user sessions
            del self.user_sessions[user_id]

            logger.info(f"Cleaned up session for user {user_id}")

    async def authenticate_connection(self, websocket: WebSocketServerProtocol, token: str) -> Optional[Dict]:
        """Authenticate WebSocket connection"""
        try:
            user_info = await verify_websocket_token(token)
            return user_info
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None

    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket connection lifecycle"""
        await self.register_connection(websocket)

        try:
            # Wait for authentication message
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)

            if auth_data.get('type') != 'authenticate':
                await websocket.close(code=4000, reason='Authentication required')
                return

            # Authenticate user
            user_info = await self.authenticate_connection(websocket, auth_data.get('token'))
            if not user_info:
                await websocket.close(code=4001, reason='Authentication failed')
                return

            # Store user session
            session_id = auth_data.get('session_id')
            document_id = auth_data.get('document_id')

            self.user_sessions[user_info['user_id']] = {
                'websocket': websocket,
                'user_info': user_info,
                'session_id': session_id,
                'document_id': document_id,
                'connected_at': datetime.utcnow()
            }

            # Add to document sessions
            if document_id:
                if document_id not in self.document_sessions:
                    self.document_sessions[document_id] = set()
                self.document_sessions[document_id].add(user_info['user_id'])

            # Initialize with manager
            await self.manager.connect(websocket, session_id, user_info)

            # Send connection success
            await websocket.send(json.dumps({
                'type': 'connection_established',
                'user_id': user_info['user_id'],
                'session_id': session_id,
                'timestamp': datetime.utcnow().isoformat()
            }))

            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    self.stats['messages_processed'] += 1
                    await self.manager.handle_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                    self.stats['errors_handled'] += 1
                    await websocket.send(json.dumps({
                        'type': MessageType.ERROR.value,
                        'error': 'Invalid JSON format',
                        'timestamp': datetime.utcnow().isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    self.stats['errors_handled'] += 1
                    await websocket.send(json.dumps({
                        'type': MessageType.ERROR.value,
                        'error': 'Internal server error',
                        'timestamp': datetime.utcnow().isoformat()
                    }))

        except ConnectionClosedOK:
            logger.info("Connection closed normally")
        except ConnectionClosedError as e:
            logger.error(f"Connection closed with error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in connection handler: {e}")
        finally:
            await self.unregister_connection(websocket)

    async def broadcast_to_document(self, document_id: str, message: Dict, exclude_user: str = None):
        """Broadcast message to all users in a document"""
        if document_id not in self.document_sessions:
            return

        for user_id in self.document_sessions[document_id]:
            if exclude_user and user_id == exclude_user:
                continue

            if user_id in self.user_sessions:
                websocket = self.user_sessions[user_id]['websocket']
                try:
                    await websocket.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to user {user_id}: {e}")

    async def broadcast_to_session(self, session_id: str, message: Dict, exclude_user: str = None):
        """Broadcast message to all users in a session"""
        for user_id, session_info in self.user_sessions.items():
            if session_info['session_id'] == session_id:
                if exclude_user and user_id == exclude_user:
                    continue

                websocket = session_info['websocket']
                try:
                    await websocket.send(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to user {user_id}: {e}")

    async def send_system_notification(self, message: str, level: str = "info"):
        """Send system notification to all connected users"""
        notification = {
            'type': MessageType.NOTIFICATION.value,
            'notification': {
                'message': message,
                'level': level,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        for websocket in self.active_connections:
            try:
                await websocket.send(json.dumps(notification))
            except Exception as e:
                logger.error(f"Error sending system notification: {e}")

    async def get_server_stats(self) -> Dict:
        """Get server statistics"""
        manager_stats = await self.manager.get_connection_stats()
        return {
            **self.stats,
            **manager_stats,
            'uptime': 'N/A',  # Would need to track start time
            'document_sessions': len(self.document_sessions),
            'user_sessions': len(self.user_sessions)
        }

    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Check if we can access Redis
            await self.manager.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {config.WS_HOST}:{config.WS_PORT}")

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for signal_name in ['SIGINT', 'SIGTERM']:
            loop.add_signal_handler(
                getattr(signal, signal_name),
                lambda: asyncio.create_task(self.shutdown())
            )

        # Start the server
        async with websockets.serve(
            self.handle_connection,
            config.WS_HOST,
            config.WS_PORT,
            ping_interval=30,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB max message size
            compression="deflate"
        ):
            logger.info("WebSocket server started successfully")
            await asyncio.Future()  # Run forever

    async def shutdown(self):
        """Gracefully shutdown the server"""
        logger.info("Shutting down WebSocket server...")

        # Close all connections
        for websocket in self.active_connections.copy():
            try:
                await websocket.close(code=1001, reason='Server shutting down')
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

        # Wait for connections to close
        await asyncio.sleep(1)

        logger.info("WebSocket server shutdown complete")
        sys.exit(0)

# Global server instance
server = CollaborationWebSocketServer()

async def main():
    """Main entry point"""
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())