#!/usr/bin/env python3
"""
RESTful API for the collaboration system

This module provides FastAPI endpoints for managing collaboration features,
including session management, user authentication, content contributions,
and community interactions.
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import jwt
import hashlib
import secrets
from sqlalchemy import create_engine, desc, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from ..core.real_time_collaboration import (
    RealTimeCollaborationManager, UserPermission, EditOperation, EditChange
)
from ..models.collaboration_models import (
    Base, User, Document, Comment, Vote, QnAQuestion, QnAAnswer,
    StudyGroup, LiveChat, ExpertAMASession, Notification,
    CommunityContribution, UserAchievement, PullRequest, Review
)

# Initialize FastAPI app
app = FastAPI(
    title="AI Documentation Collaboration API",
    description="API for real-time collaboration features in AI documentation platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database setup
DATABASE_URL = "postgresql://user:password@localhost/ai_docs_collaboration"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize collaboration manager
collaboration_manager = RealTimeCollaborationManager()

# Pydantic models for API
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str
    expertise_areas: Optional[List[str]] = []

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    expertise_areas: List[str] = []
    reputation_score: int
    join_date: datetime
    is_active: bool
    is_verified: bool

class CommentCreate(BaseModel):
    content: str
    position: Optional[int] = None
    line_number: Optional[int] = None
    comment_type: str = "general"
    parent_comment_id: Optional[str] = None

class CommentResponse(BaseModel):
    id: str
    content: str
    position: Optional[int] = None
    line_number: Optional[int] = None
    comment_type: str
    is_resolved: bool
    created_at: datetime
    updated_at: datetime
    author: UserResponse
    replies: List['CommentResponse'] = []
    vote_count: int = 0

class SessionCreate(BaseModel):
    document_id: str
    max_participants: int = 10
    is_private: bool = False

class SessionResponse(BaseModel):
    session_id: str
    document_id: str
    participant_count: int
    created_at: datetime
    last_activity: datetime
    participants: List[UserResponse]

class QnAQuestionCreate(BaseModel):
    title: str
    content: str
    tags: Optional[List[str]] = []
    question_type: str = "general"
    difficulty_level: str = "intermediate"
    document_id: Optional[str] = None

class QnAAnswerCreate(BaseModel):
    content: str
    attachments: Optional[Dict[str, Any]] = None

class StudyGroupCreate(BaseModel):
    name: str
    description: Optional[str] = None
    topic_focus: str
    max_members: int = 20
    is_private: bool = False
    learning_goals: Optional[List[str]] = []

# Utility functions
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token and return user ID"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def get_current_user(user_id: str = Depends(verify_token), db: Session = Depends(get_db)) -> User:
    """Get current user from database"""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Documentation Collaboration API", "version": "1.0.0"}

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(
            or_(User.email == user_data.email, User.username == user_data.username)
        ).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        # Create new user
        hashed_password = hash_password(user_data.password)
        user = User(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            expertise_areas=user_data.expertise_areas,
            join_date=datetime.utcnow()
        )
        user.set_password(hashed_password)  # Assuming you have a password setter

        db.add(user)
        db.commit()
        db.refresh(user)

        return user

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Registration failed")

@app.post("/auth/login")
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """User login"""
    user = db.query(User).filter(User.email == user_credentials.email).first()
    if not user or not user.check_password(user_credentials.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

# Document collaboration endpoints
@app.post("/sessions", response_model=SessionResponse)
async def create_collaboration_session(
    session_data: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new collaboration session"""
    # Get document
    document = db.query(Document).filter(Document.id == session_data.document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Create session
    session_id = await collaboration_manager.create_session(
        document_id=session_data.document_id,
        initial_content=document.content,
        creator_id=str(current_user.id),
        creator_info={
            'username': current_user.username,
            'email': current_user.email,
            'avatar_url': current_user.avatar_url
        }
    )

    return {
        "session_id": session_id,
        "document_id": session_data.document_id,
        "participant_count": 1,
        "created_at": datetime.utcnow(),
        "last_activity": datetime.utcnow(),
        "participants": [current_user]
    }

@app.get("/sessions", response_model=List[SessionResponse])
async def get_active_sessions(current_user: User = Depends(get_current_user)):
    """Get list of active collaboration sessions"""
    sessions = collaboration_manager.get_active_sessions()
    return sessions

@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session_info(session_id: str, current_user: User = Depends(get_current_user)):
    """Get information about a specific session"""
    session_info = collaboration_manager.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_info

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time collaboration"""
    # Extract user ID from query parameters (in production, use proper auth)
    user_id = websocket.query_params.get("user_id")
    if not user_id:
        await websocket.close(code=4001)
        return

    await collaboration_manager.handle_websocket_connection(websocket, session_id, user_id)

# Comment system endpoints
@app.post("/documents/{document_id}/comments", response_model=CommentResponse)
async def create_comment(
    document_id: str,
    comment_data: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new comment on a document"""
    # Verify document exists
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Create comment
    comment = Comment(
        document_id=document_id,
        user_id=current_user.id,
        content=comment_data.content,
        position=comment_data.position,
        line_number=comment_data.line_number,
        comment_type=comment_data.comment_type,
        parent_comment_id=comment_data.parent_comment_id
    )

    db.add(comment)
    db.commit()
    db.refresh(comment)

    # Add to collaboration session if active
    await collaboration_manager._add_comment(
        document_id, str(current_user.id), {
            'text': comment_data.content,
            'position': comment_data.position or 0
        }
    )

    return comment

@app.get("/documents/{document_id}/comments", response_model=List[CommentResponse])
async def get_document_comments(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all comments for a document"""
    comments = db.query(Comment).filter(
        Comment.document_id == document_id,
        Comment.parent_comment_id.is_(None),
        Comment.is_deleted == False
    ).order_by(Comment.created_at.desc()).all()

    return comments

@app.post("/comments/{comment_id}/vote")
async def vote_comment(
    comment_id: str,
    vote_type: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Vote on a comment"""
    if vote_type not in ["up", "down"]:
        raise HTTPException(status_code=400, detail="Invalid vote type")

    # Check if user already voted
    existing_vote = db.query(Vote).filter(
        Vote.user_id == current_user.id,
        Vote.comment_id == comment_id
    ).first()

    if existing_vote:
        # Update existing vote
        existing_vote.vote_type = vote_type
    else:
        # Create new vote
        vote = Vote(
            user_id=current_user.id,
            comment_id=comment_id,
            vote_type=vote_type
        )
        db.add(vote)

    db.commit()

    # Update user reputation
    if vote_type == "up":
        current_user.reputation_score += 1
    else:
        current_user.reputation_score -= 1

    db.commit()

    return {"message": f"Comment {vote_type}voted successfully"}

# Q&A System endpoints
@app.post("/questions", response_model=QnAQuestionCreate)
async def create_question(
    question_data: QnAQuestionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new Q&A question"""
    question = QnAQuestion(
        title=question_data.title,
        content=question_data.content,
        user_id=current_user.id,
        tags=question_data.tags,
        question_type=question_data.question_type,
        difficulty_level=question_data.difficulty_level,
        document_id=question_data.document_id
    )

    db.add(question)
    db.commit()
    db.refresh(question)

    return question

@app.get("/questions", response_model=List[QnAQuestionCreate])
async def get_questions(
    skip: int = 0,
    limit: int = 20,
    tag: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get Q&A questions with optional filtering"""
    query = db.query(QnAQuestion)

    if tag:
        query = query.filter(QnAQuestion.tags.contains([tag]))

    questions = query.order_by(desc(QnAQuestion.created_at)).offset(skip).limit(limit).all()
    return questions

@app.post("/questions/{question_id}/answers", response_model=QnAAnswerCreate)
async def create_answer(
    question_id: str,
    answer_data: QnAAnswerCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create an answer to a question"""
    question = db.query(QnAQuestion).filter(QnAQuestion.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    answer = QnAAnswer(
        question_id=question_id,
        user_id=current_user.id,
        content=answer_data.content,
        attachments=answer_data.attachments
    )

    db.add(answer)
    question.answer_count += 1
    db.commit()
    db.refresh(answer)

    return answer

# Study Group endpoints
@app.post("/study-groups", response_model=StudyGroupCreate)
async def create_study_group(
    group_data: StudyGroupCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new study group"""
    group = StudyGroup(
        name=group_data.name,
        description=group_data.description,
        creator_id=current_user.id,
        topic_focus=group_data.topic_focus,
        max_members=group_data.max_members,
        is_private=group_data.is_private,
        learning_goals=group_data.learning_goals
    )

    # Generate invite code
    import random
    import string
    group.invite_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    db.add(group)
    db.commit()
    db.refresh(group)

    # Add creator as member
    db.execute(
        study_group_members.insert().values(
            group_id=group.id,
            user_id=current_user.id,
            role='admin'
        )
    )
    db.commit()

    return group

@app.get("/study-groups")
async def get_study_groups(
    topic: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get study groups, optionally filtered by topic"""
    query = db.query(StudyGroup).filter(StudyGroup.is_active == True)

    if topic:
        query = query.filter(StudyGroup.topic_focus.ilike(f"%{topic}%"))

    groups = query.order_by(desc(StudyGroup.created_at)).all()
    return groups

@app.post("/study-groups/{group_id}/join")
async def join_study_group(
    group_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Join a study group"""
    group = db.query(StudyGroup).filter(StudyGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Study group not found")

    # Check if already a member
    existing_member = db.execute(
        study_group_members.select().where(
            and_(
                study_group_members.c.group_id == group_id,
                study_group_members.c.user_id == current_user.id
            )
        )
    ).fetchone()

    if existing_member:
        raise HTTPException(status_code=400, detail="Already a member of this group")

    # Check group capacity
    member_count = db.execute(
        study_group_members.select().where(
            and_(
                study_group_members.c.group_id == group_id,
                study_group_members.c.is_active == True
            )
        )
    ).fetchall()

    if len(member_count) >= group.max_members:
        raise HTTPException(status_code=400, detail="Study group is full")

    # Add member
    db.execute(
        study_group_members.insert().values(
            group_id=group_id,
            user_id=current_user.id,
            role='member'
        )
    )
    db.commit()

    return {"message": "Successfully joined study group"}

# Community contribution endpoints
@app.post("/contributions")
async def create_contribution(
    contribution_type: str,
    document_id: str,
    contribution_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a community contribution"""
    contribution = CommunityContribution(
        user_id=current_user.id,
        contribution_type=contribution_type,
        document_id=document_id,
        contribution_data=contribution_data
    )

    db.add(contribution)
    db.commit()

    # Update user reputation
    current_user.reputation_score += 5
    db.commit()

    return {"message": "Contribution created successfully"}

@app.get("/users/{user_id}/contributions")
async def get_user_contributions(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's contributions"""
    contributions = db.query(CommunityContribution).filter(
        CommunityContribution.user_id == user_id
    ).order_by(desc(CommunityContribution.created_at)).all()

    return contributions

# Notification endpoints
@app.get("/notifications")
async def get_notifications(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user notifications"""
    notifications = db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_dismissed == False
    ).order_by(desc(Notification.created_at)).limit(50).all()

    return notifications

@app.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark notification as read"""
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()

    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    notification.is_read = True
    notification.read_at = datetime.utcnow()
    db.commit()

    return {"message": "Notification marked as read"}

# Pull Request endpoints
@app.post("/pull-requests")
async def create_pull_request(
    title: str,
    description: str,
    source_document_id: str,
    target_document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a pull request"""
    pr = PullRequest(
        title=title,
        description=description,
        author_id=current_user.id,
        source_document_id=source_document_id,
        target_document_id=target_document_id
    )

    db.add(pr)
    db.commit()
    db.refresh(pr)

    return pr

@app.get("/pull-requests")
async def get_pull_requests(
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get pull requests"""
    query = db.query(PullRequest)

    if status:
        query = query.filter(PullRequest.status == status)

    prs = query.order_by(desc(PullRequest.created_at)).all()
    return prs

# Live Chat endpoints
@app.post("/live-chat")
async def send_chat_message(
    message: str,
    document_id: Optional[str] = None,
    session_id: Optional[str] = None,
    message_type: str = "text",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a live chat message"""
    chat = LiveChat(
        user_id=current_user.id,
        message=message,
        document_id=document_id,
        session_id=session_id,
        message_type=message_type
    )

    db.add(chat)
    db.commit()
    db.refresh(chat)

    # Broadcast to WebSocket if session exists
    if session_id:
        await collaboration_manager._broadcast_to_session(session_id, {
            'type': 'chat_message',
            'message': {
                'id': str(chat.id),
                'user_id': str(current_user.id),
                'username': current_user.username,
                'message': message,
                'timestamp': chat.created_at.isoformat()
            }
        })

    return chat

@app.get("/live-chat/{document_id}")
async def get_chat_history(
    document_id: str,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get chat history for a document"""
    chats = db.query(LiveChat).filter(
        LiveChat.document_id == document_id
    ).order_by(desc(LiveChat.created_at)).limit(limit).all()

    return chats

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)