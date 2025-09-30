#!/usr/bin/env python3
"""
Database models for the collaboration system

This module defines the data models for storing collaboration data,
including user sessions, edit history, comments, and community features.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Table, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

Base = declarative_base()

# Association table for many-to-many relationship between users and documents
document_contributors = Table(
    'document_contributors',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id'), primary_key=True),
    Column('document_id', UUID(as_uuid=True), ForeignKey('documents.id'), primary_key=True),
    Column('permission_level', String(50), default='editor'),
    Column('joined_at', DateTime, default=datetime.utcnow)
)

class User(Base):
    """User model for collaboration platform"""
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=True)
    avatar_url = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    expertise_areas = Column(JSONB, nullable=True)  # List of AI topics user is expert in
    reputation_score = Column(Integer, default=0)
    join_date = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    preferences = Column(JSONB, nullable=True)  # User preferences for collaboration
    notification_settings = Column(JSONB, nullable=True)

    # Relationships
    contributed_documents = relationship("Document", secondary=document_contributors, back_populates="contributors")
    created_documents = relationship("Document", back_populates="creator")
    comments = relationship("Comment", back_populates="author")
    votes = relationship("Vote", back_populates="user")
    sessions = relationship("CollaborationSession", back_populates="participants")
    study_groups = relationship("StudyGroup", back_populates="members")
    qna_questions = relationship("QnAQuestion", back_populates="author")
    qna_answers = relationship("QnAAnswer", back_populates="author")

class Document(Base):
    """Document model for AI documentation"""
    __tablename__ = 'documents'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    section_id = Column(String(50), nullable=False)  # Section identifier (e.g., "01_Foundational_Machine_Learning")
    topic_path = Column(String(500), nullable=False)  # Full path to document
    version = Column(Integer, default=1)
    creator_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_published = Column(Boolean, default=True)
    view_count = Column(Integer, default=0)
    edit_count = Column(Integer, default=0)
    collaboration_settings = Column(JSONB, nullable=True)
    tags = Column(JSONB, nullable=True)  # Tags for categorization
    difficulty_level = Column(String(20), default='intermediate')  # beginner, intermediate, advanced, expert
    estimated_read_time = Column(Integer, nullable=True)  # Minutes

    # Relationships
    creator = relationship("User", back_populates="created_documents")
    contributors = relationship("User", secondary=document_contributors, back_populates="contributed_documents")
    comments = relationship("Comment", back_populates="document")
    edit_history = relationship("EditHistory", back_populates="document")
    sessions = relationship("CollaborationSession", back_populates="document")
    qna_questions = relationship("QnAQuestion", back_populates="document")

class EditHistory(Base):
    """Edit history tracking for documents"""
    __tablename__ = 'edit_history'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    change_type = Column(String(50), nullable=False)  # insert, delete, replace, annotate
    position = Column(Integer, nullable=False)
    old_text = Column(Text, nullable=True)
    new_text = Column(Text, nullable=True)
    version_before = Column(Integer, nullable=False)
    version_after = Column(Integer, nullable=False)
    change_metadata = Column(JSONB, nullable=True)  # Additional metadata about the change
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_reverted = Column(Boolean, default=False)
    reverted_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    revert_reason = Column(Text, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="edit_history")
    user = relationship("User")

class Comment(Base):
    """Comment model for document discussions"""
    __tablename__ = 'comments'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    parent_comment_id = Column(UUID(as_uuid=True), ForeignKey('comments.id'), nullable=True)
    content = Column(Text, nullable=False)
    position = Column(Integer, nullable=True)  # Position in document for inline comments
    line_number = Column(Integer, nullable=True)
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted = Column(Boolean, default=False)
    comment_type = Column(String(20), default='general')  # general, suggestion, question, correction
    attachments = Column(JSONB, nullable=True)  # List of attachment URLs

    # Relationships
    document = relationship("Document", back_populates="comments")
    author = relationship("User", back_populates="comments")
    replies = relationship("Comment", remote_side=[id])
    parent_comment = relationship("Comment", remote_side=[id])
    votes = relationship("Vote", back_populates="comment")

class Vote(Base):
    """Vote model for community engagement"""
    __tablename__ = 'votes'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    comment_id = Column(UUID(as_uuid=True), ForeignKey('comments.id'), nullable=True)
    question_id = Column(UUID(as_uuid=True), ForeignKey('qna_questions.id'), nullable=True)
    answer_id = Column(UUID(as_uuid=True), ForeignKey('qna_answers.id'), nullable=True)
    vote_type = Column(String(10), nullable=False)  # up, down
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="votes")
    comment = relationship("Comment", back_populates="votes")

class CollaborationSession(Base):
    """Active collaboration session tracking"""
    __tablename__ = 'collaboration_sessions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'))
    session_id = Column(String(100), unique=True, nullable=False)  # WebSocket session ID
    participants = relationship("User", secondary="session_participants")
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    max_participants = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)
    session_metadata = Column(JSONB, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="sessions")

# Association table for session participants
session_participants = Table(
    'session_participants',
    Base.metadata,
    Column('session_id', UUID(as_uuid=True), ForeignKey('collaboration_sessions.id')),
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id')),
    Column('joined_at', DateTime, default=datetime.utcnow),
    Column('left_at', DateTime, nullable=True)
)

class StudyGroup(Base):
    """Study group for collaborative learning"""
    __tablename__ = 'study_groups'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    creator_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    topic_focus = Column(String(100), nullable=False)  # AI topic area
    max_members = Column(Integer, default=20)
    is_private = Column(Boolean, default=False)
    invite_code = Column(String(20), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    meeting_schedule = Column(JSONB, nullable=True)  # Recurring meeting schedule
    learning_goals = Column(JSONB, nullable=True)  # List of learning objectives

    # Relationships
    creator = relationship("User", back_populates="study_groups")
    members = relationship("User", secondary="study_group_members")

# Association table for study group members
study_group_members = Table(
    'study_group_members',
    Base.metadata,
    Column('group_id', UUID(as_uuid=True), ForeignKey('study_groups.id')),
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id')),
    Column('joined_at', DateTime, default=datetime.utcnow),
    Column('role', String(20), default='member'),  # member, moderator, admin
    Column('is_active', Column(Boolean, default=True))
)

class QnAQuestion(Base):
    """Question and Answer system for community support"""
    __tablename__ = 'qna_questions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(JSONB, nullable=True)
    view_count = Column(Integer, default=0)
    answer_count = Column(Integer, default=0)
    is_answered = Column(Boolean, default=False)
    is_pinned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    question_type = Column(String(20), default='general')  # general, technical, clarification, suggestion
    difficulty_level = Column(String(20), default='intermediate')
    accepted_answer_id = Column(UUID(as_uuid=True), ForeignKey('qna_answers.id'), nullable=True)

    # Relationships
    author = relationship("User", back_populates="qna_questions")
    document = relationship("Document", back_populates="qna_questions")
    answers = relationship("QnAAnswer", back_populates="question")

class QnAAnswer(Base):
    """Answers for Q&A questions"""
    __tablename__ = 'qna_answers'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_id = Column(UUID(as_uuid=True), ForeignKey('qna_questions.id'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    content = Column(Text, nullable=False)
    is_accepted = Column(Boolean, default=False)
    upvote_count = Column(Integer, default=0)
    downvote_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    attachments = Column(JSONB, nullable=True)  # Code snippets, images, etc.

    # Relationships
    question = relationship("QnAQuestion", back_populates="answers")
    author = relationship("User", back_populates="qna_answers")

class LiveChat(Base):
    """Live chat for real-time discussions"""
    __tablename__ = 'live_chats'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=True)
    session_id = Column(String(100), nullable=True)  # Associated collaboration session
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    message = Column(Text, nullable=False)
    message_type = Column(String(20), default='text')  # text, code, image, system
    is_edited = Column(Boolean, default=False)
    edited_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    reply_to_id = Column(UUID(as_uuid=True), ForeignKey('live_chats.id'), nullable=True)

    # Relationships
    user = relationship("User")
    replies = relationship("LiveChat", remote_side=[id])

class ExpertAMASession(Base):
    """Expert AMA (Ask Me Anything) sessions"""
    __tablename__ = 'expert_ama_sessions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    expert_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    topic_focus = Column(String(100), nullable=False)
    scheduled_start = Column(DateTime, nullable=False)
    scheduled_end = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=False)
    max_participants = Column(Integer, default=100)
    question_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    expert = relationship("User")

class Notification(Base):
    """User notification system"""
    __tablename__ = 'notifications'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    notification_type = Column(String(50), nullable=False)  # comment, mention, vote, session_invite, etc.
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSONB, nullable=True)  # Additional notification data
    is_read = Column(Boolean, default=False)
    is_dismissed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User")

class CommunityContribution(Base):
    """Track community contributions and achievements"""
    __tablename__ = 'community_contributions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    contribution_type = Column(String(50), nullable=False)  # edit, comment, answer, review, etc.
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=True)
    contribution_data = Column(JSONB, nullable=True)  # Data about the contribution
    quality_score = Column(Float, nullable=True)  # AI-evaluated quality score
    community_score = Column(Integer, default=0)  # Community-voted score
    is_featured = Column(Boolean, default=False)
    is_approved = Column(Boolean, default=True)  # For moderated contributions
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User")
    document = relationship("Document")

class UserAchievement(Base):
    """User achievements and badges"""
    __tablename__ = 'user_achievements'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    achievement_type = Column(String(50), nullable=False)  # first_edit, helpful_comment, etc.
    achievement_name = Column(String(100), nullable=False)
    achievement_description = Column(Text, nullable=True)
    badge_icon = Column(String(100), nullable=True)
    earned_at = Column(DateTime, default=datetime.utcnow)
    achievement_level = Column(Integer, default=1)  # 1-5 for tiered achievements

    # Relationships
    user = relationship("User")

class PullRequest(Base):
    """Pull request system for content contributions"""
    __tablename__ = 'pull_requests'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    author_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    source_document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'))
    target_document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'))
    status = Column(String(20), default='open')  # open, merged, closed
    review_count = Column(Integer, default=0)
    approved_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    merged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    author = relationship("User")

class Review(Base):
    """Code/content review system"""
    __tablename__ = 'reviews'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pull_request_id = Column(UUID(as_uuid=True), ForeignKey('pull_requests.id'))
    reviewer_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    review_type = Column(String(20), default='general')  # general, technical, style, security
    status = Column(String(20), default='pending')  # pending, approved, changes_requested
    comments = Column(JSONB, nullable=True)  # List of review comments
    score = Column(Integer, nullable=True)  # 1-5 quality score
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    pull_request = relationship("PullRequest")
    reviewer = relationship("User")