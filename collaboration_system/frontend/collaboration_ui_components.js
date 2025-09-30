/**
 * Real-time Collaboration UI Components
 *
 * This module provides React/JavaScript components for the collaboration interface,
 * including real-time editing, user presence, chat, and community features.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { WebSocketProvider, useWebSocket } from './websocket';
import { AuthProvider, useAuth } from './auth';
import { marked } from 'marked';
import Prism from 'prismjs';
import 'prismjs/themes/prism.css';

// Configuration
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8001';
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Utility functions
const generateId = () => Math.random().toString(36).substr(2, 9);
const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

// Real-time Document Editor Component
export const CollaborativeEditor = ({ documentId, initialContent, onContentChange }) => {
  const { socket, isConnected, lastMessage } = useWebSocket();
  const { user } = useAuth();
  const [content, setContent] = useState(initialContent || '');
  const [participants, setParticipants] = useState([]);
  const [remoteCursors, setRemoteCursors] = useState({});
  const [isEditing, setIsEditing] = useState(false);
  const [version, setVersion] = useState(1);
  const editorRef = useRef(null);
  const editQueue = useRef([]);
  const isApplyingRemoteEdit = useRef(false);

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    const data = JSON.parse(lastMessage.data);

    switch (data.type) {
      case 'document_state':
        if (!isEditing) {
          setContent(data.document.content);
          setParticipants(data.document.participants);
        }
        break;

      case 'edit_applied':
        handleRemoteEdit(data.edit);
        break;

      case 'user_joined':
        setParticipants(prev => [...prev, data.user]);
        break;

      case 'user_left':
        setParticipants(prev => prev.filter(p => p.user_id !== data.user_id));
        break;

      case 'cursor_update':
        setRemoteCursors(prev => ({
          ...prev,
          [data.user_id]: {
            ...data.cursor,
            username: data.username,
            timestamp: data.timestamp
          }
        }));
        break;

      case 'typing_indicator':
        setParticipants(prev =>
          prev.map(p =>
            p.user_id === data.user_id
              ? { ...p, is_typing: data.is_typing }
              : p
          )
        );
        break;
    }
  }, [lastMessage]);

  // Join collaboration session
  useEffect(() => {
    if (socket && documentId && user) {
      socket.send(JSON.stringify({
        type: 'join_session',
        document_id: documentId,
        user_id: user.id,
        user_info: {
          username: user.username,
          email: user.email,
          avatar_url: user.avatar_url,
          permissions: user.permissions
        }
      }));
    }
  }, [socket, documentId, user]);

  // Handle local edits
  const handleLocalEdit = useCallback((newContent, change) => {
    if (isApplyingRemoteEdit.current) return;

    setContent(newContent);
    setIsEditing(true);

    // Queue edit for sending
    editQueue.current.push({
      type: 'edit',
      edit: {
        operation: change.type,
        position: change.position,
        text: change.text,
        timestamp: Date.now()
      }
    });

    // Send edit with debouncing
    debouncedSendEdits();
  }, []);

  const debouncedSendEdits = useCallback(debounce(() => {
    if (socket && editQueue.current.length > 0) {
      editQueue.current.forEach(edit => {
        socket.send(JSON.stringify(edit));
      });
      editQueue.current = [];
      setIsEditing(false);
    }
  }, 100), [socket]);

  // Handle remote edits
  const handleRemoteEdit = (edit) => {
    isApplyingRemoteEdit.current = true;

    try {
      const { operation, position, text } = edit;
      const currentContent = content;

      let newContent;
      switch (operation) {
        case 'insert':
          newContent =
            currentContent.slice(0, position) +
            text +
            currentContent.slice(position);
          break;
        case 'delete':
          newContent =
            currentContent.slice(0, position) +
            currentContent.slice(position + text.length);
          break;
        case 'replace':
          newContent =
            currentContent.slice(0, position) +
            text +
            currentContent.slice(position + text.length);
          break;
        default:
          newContent = currentContent;
      }

      setContent(newContent);
      setVersion(prev => prev + 1);

      if (onContentChange) {
        onContentChange(newContent);
      }
    } finally {
      isApplyingRemoteEdit.current = false;
    }
  };

  // Handle cursor position updates
  const handleCursorUpdate = useCallback((position, selection) => {
    if (socket && user) {
      socket.send(JSON.stringify({
        type: 'cursor_update',
        cursor: {
          line: position.line,
          column: position.column,
          selection_start: selection.start,
          selection_end: selection.end
        }
      }));
    }
  }, [socket, user]);

  // Handle typing indicator
  const handleTyping = useCallback(() => {
    if (socket && user) {
      socket.send(JSON.stringify({
        type: 'typing_indicator',
        is_typing: true
      }));

      setTimeout(() => {
        if (socket) {
          socket.send(JSON.stringify({
            type: 'typing_indicator',
            is_typing: false
          }));
        }
      }, 1000);
    }
  }, [socket, user]);

  return (
    <div className="collaborative-editor">
      <div className="editor-header">
        <div className="participants-list">
          {participants.map(participant => (
            <div key={participant.user_id} className="participant">
              <img
                src={participant.avatar_url || '/default-avatar.png'}
                alt={participant.username}
                className="participant-avatar"
              />
              <span className={`participant-status ${participant.is_typing ? 'typing' : ''}`}>
                {participant.username}
                {participant.is_typing && ' is typing...'}
              </span>
            </div>
          ))}
        </div>
        <div className="connection-status">
          <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
          <span className="version-info">Version {version}</span>
        </div>
      </div>

      <div className="editor-container">
        <textarea
          ref={editorRef}
          value={content}
          onChange={(e) => handleLocalEdit(e.target.value, {
            type: 'replace',
            position: e.target.selectionStart,
            text: e.target.value
          })}
          onSelect={(e) => handleCursorUpdate(
            { line: 0, column: e.target.selectionStart },
            { start: e.target.selectionStart, end: e.target.selectionEnd }
          )}
          onKeyPress={handleTyping}
          className="editor-textarea"
          placeholder="Start collaborating on this document..."
        />

        {/* Remote cursors */}
        {Object.values(remoteCursors).map(cursor => (
          <div
            key={cursor.username}
            className="remote-cursor"
            style={{
              position: 'absolute',
              left: `${cursor.column * 8}px`,
              top: `${cursor.line * 20}px`
            }}
          >
            <div className="cursor-line"></div>
            <div className="cursor-label">{cursor.username}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Real-time Chat Component
export const CollaborativeChat = ({ documentId, sessionId }) => {
  const { socket, isConnected } = useWebSocket();
  const { user } = useAuth();
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const messagesEndRef = useRef(null);

  // Handle incoming chat messages
  useEffect(() => {
    if (!socket) return;

    const handleChatMessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'chat_message') {
        setMessages(prev => [...prev, data.chat]);
      }
    };

    socket.addEventListener('message', handleChatMessage);
    return () => socket.removeEventListener('message', handleChatMessage);
  }, [socket]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send message
  const sendMessage = () => {
    if (!socket || !newMessage.trim() || !user) return;

    const message = {
      type: 'chat_message',
      chat: {
        message: newMessage.trim(),
        message_type: 'text',
        document_id: documentId,
        session_id: sessionId
      }
    };

    socket.send(JSON.stringify(message));
    setNewMessage('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="collaborative-chat">
      <div className="chat-header">
        <h3>Collaboration Chat</h3>
        <span className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'Online' : 'Offline'}
        </span>
      </div>

      <div className="chat-messages">
        {messages.map(message => (
          <div
            key={message.id}
            className={`chat-message ${message.user_id === user?.id ? 'own' : 'other'}`}
          >
            <div className="message-header">
              <img
                src={message.avatar_url || '/default-avatar.png'}
                alt={message.username}
                className="message-avatar"
              />
              <div className="message-meta">
                <span className="message-author">{message.username}</span>
                <span className="message-time">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
            <div className="message-content">
              {message.message_type === 'code' ? (
                <pre><code>{message.message}</code></pre>
              ) : (
                <p>{message.message}</p>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <textarea
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          rows={3}
        />
        <button onClick={sendMessage} disabled={!newMessage.trim() || !isConnected}>
          Send
        </button>
      </div>
    </div>
  );
};

// Comment System Component
export const CommentSystem = ({ documentId }) => {
  const { user } = useAuth();
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState('');
  const [selectedText, setSelectedText] = useState('');
  const [replyingTo, setReplyingTo] = useState(null);

  // Fetch comments
  useEffect(() => {
    const fetchComments = async () => {
      try {
        const response = await fetch(`${API_URL}/documents/${documentId}/comments`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        const data = await response.json();
        setComments(data);
      } catch (error) {
        console.error('Error fetching comments:', error);
      }
    };

    fetchComments();
  }, [documentId]);

  // Handle text selection for inline comments
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      if (selection.toString().trim()) {
        setSelectedText(selection.toString().trim());
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  // Add comment
  const addComment = async () => {
    if (!newComment.trim() || !user) return;

    try {
      const response = await fetch(`${API_URL}/documents/${documentId}/comments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          content: newComment,
          position: selectedText ? 0 : null, // Simplified position
          line_number: null,
          comment_type: selectedText ? 'inline' : 'general',
          parent_comment_id: replyingTo
        })
      });

      if (response.ok) {
        const comment = await response.json();
        setComments(prev => [...prev, comment]);
        setNewComment('');
        setSelectedText('');
        setReplyingTo(null);
      }
    } catch (error) {
      console.error('Error adding comment:', error);
    }
  };

  // Vote on comment
  const voteComment = async (commentId, voteType) => {
    try {
      await fetch(`${API_URL}/comments/${commentId}/vote`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ vote_type: voteType })
      });

      // Update local state
      setComments(prev => prev.map(comment => {
        if (comment.id === commentId) {
          return {
            ...comment,
            vote_count: comment.vote_count + (voteType === 'up' ? 1 : -1)
          };
        }
        return comment;
      }));
    } catch (error) {
      console.error('Error voting on comment:', error);
    }
  };

  return (
    <div className="comment-system">
      <div className="comment-header">
        <h3>Comments & Discussion</h3>
        {selectedText && (
          <div className="selected-text">
            <strong>Selected:</strong> {selectedText}
          </div>
        )}
      </div>

      <div className="comment-input">
        <textarea
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder={selectedText ? "Comment on selected text..." : "Add a comment..."}
          rows={4}
        />
        <div className="comment-actions">
          <button onClick={addComment} disabled={!newComment.trim()}>
            {selectedText ? 'Comment on Selection' : 'Add Comment'}
          </button>
          {selectedText && (
            <button onClick={() => setSelectedText('')}>
              Clear Selection
            </button>
          )}
        </div>
      </div>

      <div className="comments-list">
        {comments.map(comment => (
          <CommentItem
            key={comment.id}
            comment={comment}
            onVote={voteComment}
            onReply={(commentId) => setReplyingTo(commentId)}
            user={user}
          />
        ))}
      </div>
    </div>
  );
};

// Individual Comment Item Component
const CommentItem = ({ comment, onVote, onReply, user }) => {
  const [showReply, setShowReply] = useState(false);
  const [replyText, setReplyText] = useState('');

  const handleReply = async () => {
    if (!replyText.trim()) return;
    // Implement reply logic
    setReplyText('');
    setShowReply(false);
  };

  return (
    <div className={`comment-item ${comment.is_resolved ? 'resolved' : ''}`}>
      <div className="comment-header">
        <img
          src={comment.author.avatar_url || '/default-avatar.png'}
          alt={comment.author.username}
          className="comment-avatar"
        />
        <div className="comment-meta">
          <span className="comment-author">{comment.author.username}</span>
          <span className="comment-time">
            {new Date(comment.created_at).toLocaleString()}
          </span>
          <span className="comment-type">{comment.comment_type}</span>
        </div>
      </div>

      <div className="comment-content">
        <p>{comment.content}</p>
        {comment.position !== null && (
          <div className="comment-position">
            Line {comment.line_number || comment.position}
          </div>
        )}
      </div>

      <div className="comment-actions">
        <button onClick={() => onVote(comment.id, 'up')}>
          ▲ {comment.vote_count || 0}
        </button>
        <button onClick={() => onVote(comment.id, 'down')}>
          ▼
        </button>
        <button onClick={() => onReply(comment.id)}>
          Reply
        </button>
        {comment.comment_type === 'suggestion' && (
          <button onClick={() => {/* Implement accept suggestion */}}>
            Accept
          </button>
        )}
      </div>

      {showReply && (
        <div className="reply-input">
          <textarea
            value={replyText}
            onChange={(e) => setReplyText(e.target.value)}
            placeholder="Write a reply..."
            rows={3}
          />
          <button onClick={handleReply}>Reply</button>
          <button onClick={() => setShowReply(false)}>Cancel</button>
        </div>
      )}

      <div className="comment-replies">
        {comment.replies && comment.replies.map(reply => (
          <CommentItem
            key={reply.id}
            comment={reply}
            onVote={onVote}
            onReply={onReply}
            user={user}
          />
        ))}
      </div>
    </div>
  );
};

// Q&A System Component
export const QnASection = ({ documentId }) => {
  const { user } = useAuth();
  const [questions, setQuestions] = useState([]);
  const [showNewQuestion, setShowNewQuestion] = useState(false);
  const [newQuestion, setNewQuestion] = useState({
    title: '',
    content: '',
    tags: [],
    difficulty_level: 'intermediate'
  });

  // Fetch questions
  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        const response = await fetch(`${API_URL}/questions?document_id=${documentId}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        const data = await response.json();
        setQuestions(data);
      } catch (error) {
        console.error('Error fetching questions:', error);
      }
    };

    fetchQuestions();
  }, [documentId]);

  // Submit new question
  const submitQuestion = async () => {
    try {
      const response = await fetch(`${API_URL}/questions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          ...newQuestion,
          document_id: documentId
        })
      });

      if (response.ok) {
        const question = await response.json();
        setQuestions(prev => [question, ...prev]);
        setNewQuestion({
          title: '',
          content: '',
          tags: [],
          difficulty_level: 'intermediate'
        });
        setShowNewQuestion(false);
      }
    } catch (error) {
      console.error('Error submitting question:', error);
    }
  };

  return (
    <div className="qna-section">
      <div className="qna-header">
        <h3>Questions & Answers</h3>
        <button onClick={() => setShowNewQuestion(!showNewQuestion)}>
          {showNewQuestion ? 'Cancel' : 'Ask Question'}
        </button>
      </div>

      {showNewQuestion && (
        <div className="new-question-form">
          <input
            type="text"
            placeholder="Question title..."
            value={newQuestion.title}
            onChange={(e) => setNewQuestion(prev => ({ ...prev, title: e.target.value }))}
          />
          <textarea
            placeholder="Describe your question in detail..."
            value={newQuestion.content}
            onChange={(e) => setNewQuestion(prev => ({ ...prev, content: e.target.value }))}
            rows={4}
          />
          <div className="form-controls">
            <select
              value={newQuestion.difficulty_level}
              onChange={(e) => setNewQuestion(prev => ({ ...prev, difficulty_level: e.target.value }))}
            >
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
              <option value="expert">Expert</option>
            </select>
            <button onClick={submitQuestion} disabled={!newQuestion.title || !newQuestion.content}>
              Submit Question
            </button>
          </div>
        </div>
      )}

      <div className="questions-list">
        {questions.map(question => (
          <QuestionItem key={question.id} question={question} user={user} />
        ))}
      </div>
    </div>
  );
};

// Individual Question Item Component
const QuestionItem = ({ question, user }) => {
  const [showAnswers, setShowAnswers] = useState(false);
  const [newAnswer, setNewAnswer] = useState('');
  const [answers, setAnswers] = useState([]);

  // Fetch answers
  useEffect(() => {
    const fetchAnswers = async () => {
      try {
        const response = await fetch(`${API_URL}/questions/${question.id}/answers`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        const data = await response.json();
        setAnswers(data);
      } catch (error) {
        console.error('Error fetching answers:', error);
      }
    };

    if (showAnswers) {
      fetchAnswers();
    }
  }, [question.id, showAnswers]);

  // Submit answer
  const submitAnswer = async () => {
    try {
      const response = await fetch(`${API_URL}/questions/${question.id}/answers`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          content: newAnswer
        })
      });

      if (response.ok) {
        const answer = await response.json();
        setAnswers(prev => [...prev, answer]);
        setNewAnswer('');
      }
    } catch (error) {
      console.error('Error submitting answer:', error);
    }
  };

  return (
    <div className="question-item">
      <div className="question-header">
        <div className="question-stats">
          <span className="answer-count">{question.answer_count} answers</span>
          <span className="view-count">{question.view_count} views</span>
          <span className={`difficulty ${question.difficulty_level}`}>
            {question.difficulty_level}
          </span>
        </div>
        <div className="question-title">{question.title}</div>
        <div className="question-meta">
          <span className="author">{question.author.username}</span>
          <span className="time">
            {new Date(question.created_at).toLocaleString()}
          </span>
        </div>
      </div>

      <div className="question-content">
        <p>{question.content}</p>
        <div className="question-tags">
          {question.tags.map(tag => (
            <span key={tag} className="tag">{tag}</span>
          ))}
        </div>
      </div>

      <div className="question-actions">
        <button onClick={() => setShowAnswers(!showAnswers)}>
          {showAnswers ? 'Hide Answers' : 'Show Answers'}
        </button>
        {question.is_answered && (
          <span className="answered-badge">✓ Answered</span>
        )}
      </div>

      {showAnswers && (
        <div className="answers-section">
          <div className="answers-list">
            {answers.map(answer => (
              <div key={answer.id} className="answer-item">
                <div className="answer-header">
                  <img
                    src={answer.author.avatar_url || '/default-avatar.png'}
                    alt={answer.author.username}
                    className="answer-avatar"
                  />
                  <div className="answer-meta">
                    <span className="author">{answer.author.username}</span>
                    <span className="time">
                      {new Date(answer.created_at).toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="answer-content">
                  <p>{answer.content}</p>
                </div>
                <div className="answer-votes">
                  <span className="score">{answer.upvote_count - answer.downvote_count}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="answer-form">
            <textarea
              value={newAnswer}
              onChange={(e) => setNewAnswer(e.target.value)}
              placeholder="Write your answer..."
              rows={4}
            />
            <button onClick={submitAnswer} disabled={!newAnswer.trim()}>
              Post Answer
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// Study Groups Component
export const StudyGroups = () => {
  const { user } = useAuth();
  const [groups, setGroups] = useState([]);
  const [showNewGroup, setShowNewGroup] = useState(false);
  const [newGroup, setNewGroup] = useState({
    name: '',
    description: '',
    topic_focus: '',
    max_members: 20,
    is_private: false,
    learning_goals: []
  });

  // Fetch study groups
  useEffect(() => {
    const fetchGroups = async () => {
      try {
        const response = await fetch(`${API_URL}/study-groups`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        const data = await response.json();
        setGroups(data);
      } catch (error) {
        console.error('Error fetching study groups:', error);
      }
    };

    fetchGroups();
  }, []);

  // Create new study group
  const createGroup = async () => {
    try {
      const response = await fetch(`${API_URL}/study-groups`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(newGroup)
      });

      if (response.ok) {
        const group = await response.json();
        setGroups(prev => [group, ...prev]);
        setNewGroup({
          name: '',
          description: '',
          topic_focus: '',
          max_members: 20,
          is_private: false,
          learning_goals: []
        });
        setShowNewGroup(false);
      }
    } catch (error) {
      console.error('Error creating study group:', error);
    }
  };

  // Join study group
  const joinGroup = async (groupId) => {
    try {
      await fetch(`${API_URL}/study-groups/${groupId}/join`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      // Refresh groups list
      const response = await fetch(`${API_URL}/study-groups`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      setGroups(data);
    } catch (error) {
      console.error('Error joining study group:', error);
    }
  };

  return (
    <div className="study-groups">
      <div className="groups-header">
        <h3>Study Groups</h3>
        <button onClick={() => setShowNewGroup(!showNewGroup)}>
          {showNewGroup ? 'Cancel' : 'Create Group'}
        </button>
      </div>

      {showNewGroup && (
        <div className="new-group-form">
          <input
            type="text"
            placeholder="Group name..."
            value={newGroup.name}
            onChange={(e) => setNewGroup(prev => ({ ...prev, name: e.target.value }))}
          />
          <textarea
            placeholder="Group description..."
            value={newGroup.description}
            onChange={(e) => setNewGroup(prev => ({ ...prev, description: e.target.value }))}
            rows={3}
          />
          <input
            type="text"
            placeholder="Topic focus..."
            value={newGroup.topic_focus}
            onChange={(e) => setNewGroup(prev => ({ ...prev, topic_focus: e.target.value }))}
          />
          <div className="form-controls">
            <label>
              <input
                type="checkbox"
                checked={newGroup.is_private}
                onChange={(e) => setNewGroup(prev => ({ ...prev, is_private: e.target.checked }))}
              />
              Private Group
            </label>
            <button onClick={createGroup} disabled={!newGroup.name || !newGroup.topic_focus}>
              Create Group
            </button>
          </div>
        </div>
      )}

      <div className="groups-grid">
        {groups.map(group => (
          <div key={group.id} className="group-card">
            <div className="group-header">
              <h4>{group.name}</h4>
              <span className={`privacy ${group.is_private ? 'private' : 'public'}`}>
                {group.is_private ? 'Private' : 'Public'}
              </span>
            </div>
            <div className="group-description">{group.description}</div>
            <div className="group-topic">
              <strong>Focus:</strong> {group.topic_focus}
            </div>
            <div className="group-stats">
              <span>Members: {group.current_members || 0}/{group.max_members}</span>
              <span>Created: {new Date(group.created_at).toLocaleDateString()}</span>
            </div>
            <div className="group-actions">
              <button onClick={() => joinGroup(group.id)}>
                Join Group
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Main Collaboration Dashboard Component
export const CollaborationDashboard = ({ documentId, documentContent }) => {
  return (
    <div className="collaboration-dashboard">
      <div className="dashboard-header">
        <h1>Collaborative AI Documentation</h1>
        <div className="dashboard-actions">
          <button>Share Document</button>
          <button>Export</button>
          <button>Settings</button>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="main-content">
          <CollaborativeEditor
            documentId={documentId}
            initialContent={documentContent}
            onContentChange={(content) => {
              // Handle content changes
            }}
          />
        </div>

        <div className="sidebar">
          <CollaborativeChat documentId={documentId} />
          <CommentSystem documentId={documentId} />
          <QnASection documentId={documentId} />
          <StudyGroups />
        </div>
      </div>
    </div>
  );
};

// CSS-in-JS Styles
const styles = `
.collaboration-dashboard {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #f5f5f5;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: white;
  border-bottom: 1px solid #ddd;
}

.dashboard-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.main-content {
  flex: 2;
  padding: 1rem;
}

.sidebar {
  flex: 1;
  padding: 1rem;
  background: white;
  border-left: 1px solid #ddd;
  overflow-y: auto;
}

.collaborative-editor {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.editor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid #ddd;
}

.participants-list {
  display: flex;
  gap: 1rem;
}

.participant {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.participant-avatar {
  width: 24px;
  height: 24px;
  border-radius: 50%;
}

.participant-status.typing::after {
  content: ' is typing...';
  color: #666;
  font-size: 0.9em;
}

.connection-status {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.status-indicator {
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.9em;
}

.status-indicator.connected {
  background: #4CAF50;
  color: white;
}

.status-indicator.disconnected {
  background: #f44336;
  color: white;
}

.editor-container {
  flex: 1;
  position: relative;
  padding: 1rem;
}

.editor-textarea {
  width: 100%;
  height: 100%;
  border: none;
  outline: none;
  font-family: 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.5;
  resize: none;
}

.remote-cursor {
  position: absolute;
  pointer-events: none;
  z-index: 10;
}

.cursor-line {
  width: 2px;
  height: 20px;
  background: #2196F3;
  animation: blink 1s infinite;
}

.cursor-label {
  position: absolute;
  top: -20px;
  left: 0;
  background: #2196F3;
  color: white;
  padding: 2px 4px;
  border-radius: 2px;
  font-size: 12px;
  white-space: nowrap;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.collaborative-chat {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  margin-bottom: 1rem;
  height: 300px;
  display: flex;
  flex-direction: column;
}

.chat-header {
  padding: 1rem;
  border-bottom: 1px solid #ddd;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.chat-message {
  margin-bottom: 1rem;
  padding: 0.5rem;
  border-radius: 4px;
}

.chat-message.own {
  background: #e3f2fd;
  margin-left: 2rem;
}

.chat-message.other {
  background: #f5f5f5;
  margin-right: 2rem;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.message-avatar {
  width: 24px;
  height: 24px;
  border-radius: 50%;
}

.message-meta {
  display: flex;
  flex-direction: column;
}

.message-author {
  font-weight: bold;
  font-size: 0.9em;
}

.message-time {
  font-size: 0.8em;
  color: #666;
}

.chat-input {
  padding: 1rem;
  border-top: 1px solid #ddd;
  display: flex;
  gap: 0.5rem;
}

.chat-input textarea {
  flex: 1;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 0.5rem;
  resize: none;
}

.chat-input button {
  padding: 0.5rem 1rem;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.chat-input button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.comment-system, .qna-section, .study-groups {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  margin-bottom: 1rem;
  padding: 1rem;
}

.groups-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.group-card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 1rem;
  background: white;
}

.group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.privacy {
  font-size: 0.8em;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
}

.privacy.private {
  background: #ffcdd2;
  color: #c62828;
}

.privacy.public {
  background: #c8e6c9;
  color: #2e7d32;
}

.group-stats {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.9em;
  color: #666;
  margin: 0.5rem 0;
}

.group-actions {
  display: flex;
  justify-content: flex-end;
}

.group-actions button {
  padding: 0.5rem 1rem;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

@media (max-width: 768px) {
  .dashboard-content {
    flex-direction: column;
  }

  .sidebar {
    border-left: none;
    border-top: 1px solid #ddd;
  }

  .groups-grid {
    grid-template-columns: 1fr;
  }
}
`;

// Inject styles
const styleSheet = document.createElement('style');
styleSheet.textContent = styles;
document.head.appendChild(styleSheet);