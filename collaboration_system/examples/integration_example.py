#!/usr/bin/env python3
"""
Integration Example: Collaboration System with AI Documentation

This example demonstrates how to integrate the real-time collaboration system
with the existing AI documentation structure, showing practical usage patterns
and implementation details.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Import collaboration system components
from core.real_time_collaboration import RealTimeCollaborationManager, EditOperation, EditChange
from api.collaboration_api import app, get_db
from models.collaboration_models import Document, User, Comment
from websocket.websocket_handlers import WebSocketManager
from utils.config import get_config
from utils.auth import create_access_token, hash_password

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDocsCollaborationIntegration:
    """
    Integration class that connects the collaboration system with AI documentation
    """

    def __init__(self):
        self.config = get_config()
        self.collaboration_manager = RealTimeCollaborationManager()
        self.websocket_manager = WebSocketManager()

        # AI Documentation paths
        self.docs_root = Path("/Users/dtumkorkmaz/Projects/ai-docs")
        self.interactive_root = self.docs_root / "interactive"
        self.sections = [
            "01_Foundational_Machine_Learning",
            "02_Advanced_Deep_Learning",
            "03_Natural_Language_Processing",
            "04_Computer_Vision",
            "05_Generative_AI",
            "06_AI_Agents_and_Autonomous",
            "07_AI_Ethics_and_Safety",
            "08_AI_Applications_Industry"
        ]

    async def setup_document_for_collaboration(self, section_name: str, document_path: str) -> Dict:
        """
        Set up a document for real-time collaboration
        """
        try:
            # Read the document content
            full_path = self.docs_root / document_path
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create document in database
            db = next(get_db())
            document = Document(
                title=f"{section_name} - Collaborative Document",
                content=content,
                section_id=section_name,
                topic_path=document_path,
                version=1,
                collaboration_settings={
                    "enable_real_time": True,
                    "max_participants": 50,
                    "enable_comments": True,
                    "enable_chat": True
                }
            )

            db.add(document)
            db.commit()
            db.refresh(document)

            logger.info(f"Document set up for collaboration: {document.id}")
            return {
                "document_id": str(document.id),
                "title": document.title,
                "section": section_name,
                "content_length": len(content),
                "collaboration_enabled": True
            }

        except Exception as e:
            logger.error(f"Error setting up document for collaboration: {e}")
            return None

    async def create_collaborative_session(self, document_id: str, user_info: Dict) -> str:
        """
        Create a new collaboration session for a document
        """
        try:
            # Get document content
            db = next(get_db())
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError("Document not found")

            # Create collaboration session
            session_id = await self.collaboration_manager.create_session(
                document_id=document_id,
                initial_content=document.content,
                creator_id=user_info['user_id'],
                creator_info=user_info
            )

            logger.info(f"Created collaboration session: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Error creating collaboration session: {e}")
            return None

    async def simulate_collaborative_editing(self, session_id: str, users: List[Dict]):
        """
        Simulate multiple users editing the same document simultaneously
        """
        logger.info(f"Starting collaborative editing simulation with {len(users)} users")

        # Users join the session
        for user in users:
            await self.collaboration_manager.join_session(session_id, user['user_id'], user)

        # Simulate concurrent edits
        edit_operations = [
            {
                "operation": EditOperation.INSERT,
                "position": 100,
                "text": "// This is a collaborative edit\n",
                "user_id": users[0]['user_id']
            },
            {
                "operation": EditOperation.INSERT,
                "position": 150,
                "text": "# Collaborative Learning Section\n",
                "user_id": users[1]['user_id']
            },
            {
                "operation": EditOperation.REPLACE,
                "position": 200,
                "text": "Updated content through collaboration",
                "user_id": users[0]['user_id']
            }
        ]

        # Apply edits with slight delays to simulate real usage
        for i, edit_data in enumerate(edit_operations):
            await asyncio.sleep(0.5)  # Simulate typing delay

            edit_change = EditChange(
                operation=edit_data['operation'],
                position=edit_data['position'],
                text=edit_data['text'],
                user_id=edit_data['user_id'],
                timestamp=datetime.utcnow(),
                version=i + 1,
                document_id=session_id
            )

            await self.collaboration_manager.apply_edit(session_id, edit_change)
            logger.info(f"Applied edit by user {edit_data['user_id']}")

        logger.info("Collaborative editing simulation completed")

    async def create_discussion_thread(self, document_id: str, user_info: Dict, comment_data: Dict):
        """
        Create a discussion thread on a specific document section
        """
        try:
            db = next(get_db())

            # Create main comment
            comment = Comment(
                document_id=document_id,
                user_id=user_info['user_id'],
                content=comment_data['content'],
                position=comment_data.get('position'),
                line_number=comment_data.get('line_number'),
                comment_type=comment_data.get('comment_type', 'general')
            )

            db.add(comment)
            db.commit()
            db.refresh(comment)

            # Broadcast to collaboration session
            await self.collaboration_manager._broadcast_to_session(
                document_id,
                {
                    'type': 'comment_added',
                    'comment': {
                        'id': str(comment.id),
                        'content': comment.content,
                        'user_id': user_info['user_id'],
                        'username': user_info['username'],
                        'timestamp': comment.created_at.isoformat()
                    }
                }
            )

            logger.info(f"Created discussion thread: {comment.id}")
            return str(comment.id)

        except Exception as e:
            logger.error(f"Error creating discussion thread: {e}")
            return None

    async def setup_study_group(self, topic: str, participants: List[Dict]) -> Dict:
        """
        Create a study group for collaborative learning
        """
        try:
            db = next(get_db())

            # Create study group (simplified version)
            from models.collaboration_models import StudyGroup

            group = StudyGroup(
                name=f"{topic} Study Group",
                description=f"Collaborative learning group for {topic}",
                creator_id=participants[0]['user_id'],
                topic_focus=topic,
                max_members=20,
                is_private=False,
                learning_goals=[
                    f"Master {topic} concepts",
                    "Complete practical exercises",
                    "Participate in discussions"
                ]
            )

            db.add(group)
            db.commit()
            db.refresh(group)

            # Add participants to group
            for participant in participants[1:]:  # Skip creator
                db.execute(
                    study_group_members.insert().values(
                        group_id=group.id,
                        user_id=participant['user_id'],
                        role='member'
                    )
                )

            db.commit()

            logger.info(f"Created study group: {group.id}")
            return {
                "group_id": str(group.id),
                "name": group.name,
                "topic": topic,
                "participant_count": len(participants)
            }

        except Exception as e:
            logger.error(f"Error setting up study group: {e}")
            return None

    async def generate_ai_suggestions(self, document_content: str, context: Dict) -> List[Dict]:
        """
        Generate AI-powered suggestions for document improvement
        """
        try:
            # This is a simplified version - in production, you'd use OpenAI API
            suggestions = []

            # Analyze content for common patterns
            if "TODO" in document_content:
                suggestions.append({
                    "type": "todo",
                    "message": "Found TODO items that need attention",
                    "priority": "high",
                    "suggestions": ["Complete missing examples", "Add documentation"]
                })

            if len(document_content) < 500:
                suggestions.append({
                    "type": "content",
                    "message": "Document is quite short",
                    "priority": "medium",
                    "suggestions": ["Add more detailed explanations", "Include code examples"]
                })

            # Check for code blocks without explanations
            if "```" in document_content:
                suggestions.append({
                    "type": "structure",
                    "message": "Code blocks detected",
                    "priority": "low",
                    "suggestions": ["Add explanations for code", "Include output examples"]
                })

            return suggestions

        except Exception as e:
            logger.error(f"Error generating AI suggestions: {e}")
            return []

    async def run_integration_demo(self):
        """
        Run a complete integration demonstration
        """
        logger.info("Starting AI Documentation Collaboration Integration Demo")

        # Sample users
        users = [
            {
                "user_id": "user_1",
                "username": "alice_ai",
                "email": "alice@example.com",
                "avatar_url": None,
                "expertise": ["machine_learning", "python"]
            },
            {
                "user_id": "user_2",
                "username": "bob_ml",
                "email": "bob@example.com",
                "avatar_url": None,
                "expertise": ["deep_learning", "tensorflow"]
            },
            {
                "user_id": "user_3",
                "username": "carol_nlp",
                "email": "carol@example.com",
                "avatar_url": None,
                "expertise": ["nlp", "transformers"]
            }
        ]

        # Step 1: Set up document for collaboration
        logger.info("Step 1: Setting up document for collaboration")
        doc_result = await self.setup_document_for_collaboration(
            "01_Foundational_Machine_Learning",
            "01_Foundational_Machine_Learning/00_Overview.md"
        )

        if not doc_result:
            logger.error("Failed to set up document")
            return

        document_id = doc_result["document_id"]
        logger.info(f"Document set up: {document_id}")

        # Step 2: Create collaboration session
        logger.info("Step 2: Creating collaboration session")
        session_id = await self.create_collaborative_session(document_id, users[0])

        if not session_id:
            logger.error("Failed to create collaboration session")
            return

        logger.info(f"Collaboration session created: {session_id}")

        # Step 3: Simulate collaborative editing
        logger.info("Step 3: Simulating collaborative editing")
        await self.simulate_collaborative_editing(session_id, users[:2])

        # Step 4: Create discussion threads
        logger.info("Step 4: Creating discussion threads")
        discussion_data = {
            "content": "Could someone explain the difference between supervised and unsupervised learning?",
            "position": 250,
            "line_number": 15,
            "comment_type": "question"
        }

        comment_id = await self.create_discussion_thread(document_id, users[1], discussion_data)
        logger.info(f"Discussion thread created: {comment_id}")

        # Step 5: Set up study group
        logger.info("Step 5: Setting up study group")
        group_result = await self.setup_study_group("Machine Learning Basics", users)
        logger.info(f"Study group created: {group_result}")

        # Step 6: Generate AI suggestions
        logger.info("Step 6: Generating AI suggestions")
        db = next(get_db())
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            suggestions = await self.generate_ai_suggestions(document.content, {})
            logger.info(f"Generated {len(suggestions)} AI suggestions")
            for suggestion in suggestions:
                logger.info(f"- {suggestion['message']}")

        # Step 7: Display session statistics
        logger.info("Step 7: Session statistics")
        session_info = self.collaboration_manager.get_session_info(session_id)
        if session_info:
            logger.info(f"Active participants: {len(session_info['participants'])}")
            logger.info(f"Edit count: {session_info['edit_count']}")
            logger.info(f"Session duration: {session_info['last_activity']}")

        logger.info("Integration demo completed successfully!")

    async def demonstrate_integration_patterns(self):
        """
        Demonstrate various integration patterns
        """
        logger.info("Demonstrating integration patterns")

        # Pattern 1: Real-time code collaboration
        await self.demonstrate_code_collaboration()

        # Pattern 2: Interactive learning sessions
        await self.demonstrate_learning_sessions()

        # Pattern 3: Community Q&A integration
        await self.demonstrate_qa_integration()

        # Pattern 4: Expert mentoring sessions
        await self.demonstrate_expert_sessions()

    async def demonstrate_code_collaboration(self):
        """
        Demonstrate real-time code collaboration
        """
        logger.info("Demonstrating code collaboration pattern")

        # Create a Python notebook for collaborative coding
        code_document = {
            "title": "Collaborative Python Notebook",
            "content": '''# Collaborative Machine Learning Notebook

## Data Loading
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data.csv')
```

## Data Preprocessing
```python
# TODO: Add preprocessing steps
```

## Model Training
```python
# TODO: Implement model training
```
''',
            "section_id": "01_Foundational_Machine_Learning",
            "topic_path": "interactive/notebooks/01_Foundational_Machine_Learning/collaborative_notebook.ipynb"
        }

        # Set up document
        doc_result = await self.setup_document_for_collaboration(
            "01_Foundational_Machine_Learning",
            "interactive/notebooks/01_Foundational_Machine_Learning/collaborative_notebook.ipynb"
        )

        if doc_result:
            logger.info("Code collaboration document created successfully")

    async def demonstrate_learning_sessions(self):
        """
        Demonstrate interactive learning sessions
        """
        logger.info("Demonstrating learning sessions pattern")

        # This would integrate with the existing interactive notebook system
        session_config = {
            "type": "learning_session",
            "topic": "Neural Networks",
            "duration_minutes": 60,
            "max_participants": 10,
            "features": [
                "real_time_code_execution",
                "collaborative_whiteboard",
                "live_chat",
                "screen_sharing"
            ]
        }

        logger.info(f"Learning session config: {session_config}")

    async def demonstrate_qa_integration(self):
        """
        Demonstrate Q&A system integration
        """
        logger.info("Demonstrating Q&A integration pattern")

        # Sample Q&A data that would integrate with documentation
        qa_examples = [
            {
                "question": "What is the difference between CNN and RNN?",
                "context": "04_Computer_Vision/00_Overview.md",
                "difficulty": "intermediate",
                "tags": ["CNN", "RNN", "deep_learning"]
            },
            {
                "question": "How do transformers handle long sequences?",
                "context": "03_Natural_Language_Processing/00_Overview.md",
                "difficulty": "advanced",
                "tags": ["transformers", "attention", "nlp"]
            }
        ]

        for qa in qa_examples:
            logger.info(f"Q&A item: {qa['question']} (Difficulty: {qa['difficulty']})")

    async def demonstrate_expert_sessions(self):
        """
        Demonstrate expert mentoring sessions
        """
        logger.info("Demonstrating expert sessions pattern")

        expert_session = {
            "expert_id": "expert_1",
            "expert_name": "Dr. AI Researcher",
            "topic": "Advanced Deep Learning",
            "schedule": "Weekly on Tuesdays at 3 PM EST",
            "max_participants": 25,
            "format": "AMA + Live Coding",
            "recording_available": True
        }

        logger.info(f"Expert session: {expert_session}")

async def main():
    """
    Main function to run the integration demonstration
    """
    integration = AIDocsCollaborationIntegration()

    try:
        # Run the full integration demo
        await integration.run_integration_demo()

        # Demonstrate various patterns
        await integration.demonstrate_integration_patterns()

        logger.info("Integration demonstration completed successfully!")

    except Exception as e:
        logger.error(f"Error in integration demo: {e}")
        raise

if __name__ == "__main__":
    # Run the integration demo
    asyncio.run(main())