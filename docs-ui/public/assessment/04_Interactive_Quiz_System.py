#!/usr/bin/env python3
"""
Interactive Quiz System
Provides dynamic quiz generation, real-time feedback, and adaptive difficulty
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from enum import Enum

class QuizMode(Enum):
    PRACTICE = "practice"          # No time limit, unlimited hints
    TIMED = "timed"               # Time limit, limited hints
    EXAM = "exam"                 # Strict time limit, no hints
    ADAPTIVE = "adaptive"         # Difficulty adjusts based on performance
    REVIEW = "review"             # Review previously incorrect questions

class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    CODING_CHALLENGE = "coding_challenge"
    ESSAY = "essay"
    DRAG_DROP = "drag_drop"
    MATCHING = "matching"
    SEQUENCE = "sequence"

@dataclass
class QuizQuestion:
    """Individual quiz question with interaction data"""
    id: str
    type: QuestionType
    content: Dict[str, Any]
    options: List[str] = None
    correct_answer: Any
    points: int
    time_limit: int = None
    hints: List[str] = None
    explanation: str = ""
    difficulty: int = 1
    section: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.hints is None:
            self.hints = []
        if self.tags is None:
            self.tags = []

@dataclass
class QuizAttempt:
    """Single quiz attempt record"""
    attempt_id: str
    user_id: str
    quiz_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    questions_answered: int = 0
    correct_answers: int = 0
    total_points: int = 0
    max_points: int = 0
    time_taken: int = 0
    answers: Dict[str, Dict] = None
    hints_used: int = 0
    mode: QuizMode = None
    completed: bool = False

    def __post_init__(self):
        if self.answers is None:
            self.answers = {}

@dataclass
class QuizSession:
    """Active quiz session management"""
    session_id: str
    user_id: str
    quiz_config: Dict[str, Any]
    questions: List[QuizQuestion]
    current_question_index: int = 0
    attempt: QuizAttempt = None
    start_time: datetime = None
    adaptive_difficulty: int = 1
    performance_history: List[float] = None
    pause_time: Optional[datetime] = None

    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []

class InteractiveQuizSystem:
    """Main interactive quiz system"""

    def __init__(self, data_path: str = "assessment_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        self.question_bank = {}
        self.quiz_templates = {}
        self.active_sessions = {}
        self.user_statistics = {}

        self.load_data()

    def load_data(self):
        """Load quiz data and templates"""
        # Load question bank
        questions_file = self.data_path / "quiz_questions.json"
        if questions_file.exists():
            with open(questions_file, 'r') as f:
                questions_data = json.load(f)
                for qid, qdata in questions_data.items():
                    self.question_bank[qid] = self._dict_to_quiz_question(qdata)

        # Load quiz templates
        templates_file = self.data_path / "quiz_templates.json"
        if templates_file.exists():
            with open(templates_file, 'r') as f:
                self.quiz_templates = json.load(f)

        # Load user statistics
        stats_file = self.data_path / "user_quiz_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.user_statistics = json.load(f)

    def save_data(self):
        """Save quiz data"""
        # Save question bank
        questions_file = self.data_path / "quiz_questions.json"
        questions_data = {qid: self._quiz_question_to_dict(q) for qid, q in self.question_bank.items()}
        with open(questions_file, 'w') as f:
            json.dump(questions_data, f, indent=2, default=str)

        # Save user statistics
        stats_file = self.data_path / "user_quiz_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.user_statistics, f, indent=2, default=str)

    def create_quiz_question(self, **kwargs) -> QuizQuestion:
        """Create a new quiz question"""
        question_id = kwargs.get('id', f"q_{int(time.time() * 1000)}")

        question = QuizQuestion(
            id=question_id,
            type=QuestionType(kwargs['type']),
            content=kwargs['content'],
            options=kwargs.get('options'),
            correct_answer=kwargs['correct_answer'],
            points=kwargs.get('points', 10),
            time_limit=kwargs.get('time_limit'),
            hints=kwargs.get('hints', []),
            explanation=kwargs.get('explanation', ''),
            difficulty=kwargs.get('difficulty', 1),
            section=kwargs.get('section', ''),
            tags=kwargs.get('tags', [])
        )

        self.question_bank[question_id] = question
        self.save_data()
        return question

    def start_quiz_session(self, user_id: str, quiz_config: Dict[str, Any]) -> str:
        """Start a new quiz session"""
        session_id = f"session_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

        # Generate questions based on configuration
        questions = self._generate_quiz_questions(quiz_config)

        # Create session
        session = QuizSession(
            session_id=session_id,
            user_id=user_id,
            quiz_config=quiz_config,
            questions=questions,
            start_time=datetime.now()
        )

        # Create attempt record
        attempt = QuizAttempt(
            attempt_id=f"attempt_{session_id}",
            user_id=user_id,
            quiz_id=quiz_config.get('quiz_id', 'custom_quiz'),
            start_time=datetime.now(),
            max_points=sum(q.points for q in questions),
            mode=QuizMode(quiz_config.get('mode', 'practice'))
        )

        session.attempt = attempt
        self.active_sessions[session_id] = session

        return session_id

    def _generate_quiz_questions(self, config: Dict[str, Any]) -> List[QuizQuestion]:
        """Generate questions for a quiz based on configuration"""
        mode = config.get('mode', 'practice')
        section = config.get('section', '')
        difficulty = config.get('difficulty', 1)
        num_questions = config.get('num_questions', 10)
        question_types = config.get('question_types', None)

        # Filter questions by criteria
        available_questions = []
        for question in self.question_bank.values():
            if section and question.section != section:
                continue
            if difficulty and question.difficulty != difficulty:
                continue
            if question_types and question.type.value not in question_types:
                continue
            available_questions.append(question)

        # If no questions match criteria, use all available
        if not available_questions:
            available_questions = list(self.question_bank.values())

        # Select questions
        if len(available_questions) <= num_questions:
            selected_questions = available_questions
        else:
            # Use weighted random selection based on difficulty and performance
            selected_questions = self._weighted_question_selection(
                available_questions, num_questions, config.get('user_id', '')
            )

        return selected_questions

    def _weighted_question_selection(self, questions: List[QuizQuestion],
                                   num_questions: int, user_id: str) -> List[QuizQuestion]:
        """Select questions using weighted random based on user performance"""
        if user_id not in self.user_statistics:
            return random.sample(questions, num_questions)

        user_stats = self.user_statistics[user_id]
        selected_questions = []

        # Calculate weights based on user's weak areas
        question_weights = []
        for question in questions:
            weight = 1.0

            # Lower weight for recently attempted questions
            recent_attempts = user_stats.get('recent_attempts', {})
            if question.id in recent_attempts:
                time_since_attempt = (datetime.now() - recent_attempts[question.id]).days
                if time_since_attempt < 7:  # Within last week
                    weight *= 0.3

            # Higher weight for weak areas
            weak_areas = user_stats.get('weak_areas', {})
            for tag in question.tags:
                if tag in weak_areas:
                    weight *= (1 + weak_areas[tag])

            question_weights.append(weight)

        # Weighted random selection
        total_weight = sum(question_weights)
        if total_weight == 0:
            return random.sample(questions, num_questions)

        probabilities = [w / total_weight for w in question_weights]
        selected_indices = np.random.choice(len(questions), size=min(num_questions, len(questions)),
                                         replace=False, p=probabilities)

        selected_questions = [questions[i] for i in selected_indices]
        return selected_questions

    def get_current_question(self, session_id: str) -> Optional[Dict]:
        """Get the current question for a quiz session"""
        session = self.active_sessions.get(session_id)
        if not session or session.current_question_index >= len(session.questions):
            return None

        question = session.questions[session.current_question_index]
        time_remaining = self._calculate_time_remaining(session, question)

        return {
            'question_id': question.id,
            'type': question.type.value,
            'content': question.content,
            'options': question.options,
            'points': question.points,
            'time_limit': question.time_limit,
            'time_remaining': time_remaining,
            'difficulty': question.difficulty,
            'hints_available': len(question.hints),
            'question_number': session.current_question_index + 1,
            'total_questions': len(session.questions),
            'progress': (session.current_question_index / len(session.questions)) * 100
        }

    def _calculate_time_remaining(self, session: QuizSession, question: QuizQuestion) -> int:
        """Calculate remaining time for current question"""
        if question.time_limit is None:
            return None

        # For adaptive mode, adjust time based on difficulty
        if session.quiz_config.get('mode') == 'adaptive':
            base_time = question.time_limit
            difficulty_adjustment = 1 + (session.adaptive_difficulty - 1) * 0.2
            adjusted_time = int(base_time * difficulty_adjustment)
        else:
            adjusted_time = question.time_limit

        # Account for session pause time
        if session.pause_time:
            pause_duration = (datetime.now() - session.pause_time).total_seconds()
            adjusted_time = max(0, adjusted_time - int(pause_duration))

        return adjusted_time

    def submit_answer(self, session_id: str, question_id: str, answer: Any,
                     time_spent: int = None) -> Dict[str, Any]:
        """Submit an answer for the current question"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {'error': 'Session not found'}

        question = session.questions[session.current_question_index]
        if question.id != question_id:
            return {'error': 'Question ID mismatch'}

        # Evaluate answer
        is_correct, score, feedback = self._evaluate_answer(question, answer)

        # Record answer
        session.attempt.answers[question_id] = {
            'answer': answer,
            'is_correct': is_correct,
            'score': score,
            'time_spent': time_spent or 0,
            'timestamp': datetime.now().isoformat()
        }

        session.attempt.questions_answered += 1
        session.attempt.total_points += score
        if is_correct:
            session.attempt.correct_answers += 1

        # Update performance history for adaptive mode
        if session.quiz_config.get('mode') == 'adaptive':
            session.performance_history.append(1 if is_correct else 0)
            self._adjust_adaptive_difficulty(session)

        # Update user statistics
        self._update_user_statistics(session.user_id, question, is_correct, time_spent or 0)

        # Move to next question
        session.current_question_index += 1

        # Check if quiz is completed
        is_completed = session.current_question_index >= len(session.questions)
        if is_completed:
            session.attempt.end_time = datetime.now()
            session.attempt.completed = True
            session.attempt.time_taken = int((session.attempt.end_time - session.attempt.start_time).total_seconds())

        return {
            'is_correct': is_correct,
            'score': score,
            'max_score': question.points,
            'feedback': feedback,
            'explanation': question.explanation,
            'is_completed': is_completed,
            'current_score': session.attempt.total_points,
            'max_possible_score': session.attempt.max_points,
            'next_question_available': not is_completed
        }

    def _evaluate_answer(self, question: QuizQuestion, answer: Any) -> Tuple[bool, int, str]:
        """Evaluate a user's answer and return correctness, score, and feedback"""

        if question.type == QuestionType.MULTIPLE_CHOICE:
            is_correct = str(answer).strip().lower() == str(question.correct_answer).strip().lower()
            score = question.points if is_correct else 0
            feedback = "Correct!" if is_correct else f"Incorrect. The correct answer is: {question.correct_answer}"

        elif question.type == QuestionType.TRUE_FALSE:
            is_correct = str(answer).strip().lower() == str(question.correct_answer).strip().lower()
            score = question.points if is_correct else 0
            feedback = "Correct!" if is_correct else f"Incorrect. The correct answer is: {question.correct_answer}"

        elif question.type == QuestionType.FILL_BLANK:
            # More flexible matching for fill-in-blank
            correct_answers = question.correct_answer if isinstance(question.correct_answer, list) else [question.correct_answer]
            is_correct = any(str(answer).strip().lower() == str(ans).strip().lower() for ans in correct_answers)
            score = question.points if is_correct else 0
            feedback = "Correct!" if is_correct else f"Incorrect. Possible answers: {', '.join(correct_answers)}"

        elif question.type == QuestionType.CODING_CHALLENGE:
            # Code evaluation (simplified)
            is_correct, score, feedback = self._evaluate_code_answer(question, answer)

        elif question.type == QuestionType.ESSAY:
            # Essay evaluation (simplified)
            is_correct, score, feedback = self._evaluate_essay_answer(question, answer)

        elif question.type == QuestionType.MATCHING:
            # Matching evaluation
            is_correct = self._evaluate_matching_answer(question, answer)
            score = question.points if is_correct else 0
            feedback = "Correct!" if is_correct else "Some matches are incorrect."

        elif question.type == QuestionType.SEQUENCE:
            # Sequence evaluation
            is_correct = answer == question.correct_answer
            score = question.points if is_correct else 0
            feedback = "Correct sequence!" if is_correct else "Incorrect sequence."

        else:
            # Default evaluation
            is_correct = str(answer) == str(question.correct_answer)
            score = question.points if is_correct else 0
            feedback = "Correct!" if is_correct else "Incorrect answer."

        return is_correct, score, feedback

    def _evaluate_code_answer(self, question: QuizQuestion, code: str) -> Tuple[bool, int, str]:
        """Evaluate coding challenge answers (simplified)"""
        # In a real implementation, this would:
        # 1. Execute the code in a sandbox
        # 2. Run test cases
        # 3. Check for code quality
        # 4. Evaluate performance

        test_cases = question.content.get('test_cases', [])
        if not test_cases:
            return False, 0, "No test cases defined"

        # Simulated evaluation based on code length and basic patterns
        code_quality_score = 0

        # Check for basic code patterns
        patterns = [
            ('def ', 'Function definition'),
            ('import ', 'Library imports'),
            ('for ', 'Loop structures'),
            ('if ', 'Conditional logic'),
            ('return ', 'Return statements'),
            ('try:', 'Error handling')
        ]

        for pattern, description in patterns:
            if pattern in code:
                code_quality_score += 1

        # Calculate score based on code quality and length
        length_score = min(1.0, len(code) / 200)  # Reasonable length bonus
        quality_score = min(1.0, code_quality_score / len(patterns))

        total_score = int(question.points * (0.5 + 0.25 * length_score + 0.25 * quality_score))
        is_correct = total_score >= question.points * 0.7

        feedback = f"Code evaluation: {code_quality_score}/{len(patterns)} patterns detected"
        if is_correct:
            feedback += ". Good work!"
        else:
            feedback += ". Consider adding more structure and error handling."

        return is_correct, total_score, feedback

    def _evaluate_essay_answer(self, question: QuizQuestion, essay: str) -> Tuple[bool, int, str]:
        """Evaluate essay answers (simplified)"""
        essay_length = len(essay.split())
        min_length = question.content.get('min_length', 50)

        if essay_length < min_length:
            return False, 0, f"Essay too short. Minimum {min_length} words required."

        # Check for key concepts
        key_concepts = question.content.get('key_concepts', [])
        covered_concepts = sum(1 for concept in key_concepts if concept.lower() in essay.lower())

        concept_score = covered_concepts / len(key_concepts) if key_concepts else 1.0
        length_score = min(1.0, essay_length / (min_length * 2))  # Bonus for longer essays

        total_score = int(question.points * (0.6 * concept_score + 0.4 * length_score))
        is_correct = total_score >= question.points * 0.7

        feedback = f"Essay covers {covered_concepts}/{len(key_concepts)} key concepts"
        if is_correct:
            feedback += ". Good understanding demonstrated."
        else:
            feedback += ". Consider including more key concepts."

        return is_correct, total_score, feedback

    def _evaluate_matching_answer(self, question: QuizQuestion, user_matches: Dict) -> bool:
        """Evaluate matching question answers"""
        correct_matches = question.correct_answer

        if not isinstance(user_matches, dict):
            return False

        # Check if all matches are correct
        for key, value in user_matches.items():
            if correct_matches.get(key) != value:
                return False

        return True

    def _adjust_adaptive_difficulty(self, session: QuizSession):
        """Adjust difficulty based on performance in adaptive mode"""
        if len(session.performance_history) < 3:
            return

        # Calculate recent performance
        recent_performance = session.performance_history[-3:]
        correct_rate = sum(recent_performance) / len(recent_performance)

        # Adjust difficulty
        if correct_rate >= 0.8:  # 80% or better
            session.adaptive_difficulty = min(5, session.adaptive_difficulty + 1)
        elif correct_rate <= 0.4:  # 40% or worse
            session.adaptive_difficulty = max(1, session.adaptive_difficulty - 1)

    def get_hint(self, session_id: str, question_id: str) -> Optional[str]:
        """Get a hint for the current question"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        question = session.questions[session.current_question_index]
        if question.id != question_id:
            return None

        # Check if hints are available
        if not question.hints:
            return "No hints available for this question."

        # Get next available hint
        hints_used = len([a for a in session.attempt.answers.values()
                         if a.get('hints_used', 0)])
        if hints_used >= len(question.hints):
            return "No more hints available."

        hint = question.hints[hints_used]

        # Record hint usage
        current_answer = session.attempt.answers.get(question_id, {})
        current_answer['hints_used'] = hints_used + 1
        session.attempt.answers[question_id] = current_answer
        session.attempt.hints_used += 1

        return hint

    def pause_session(self, session_id: str) -> bool:
        """Pause a quiz session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False

        session.pause_time = datetime.now()
        return True

    def resume_session(self, session_id: str) -> bool:
        """Resume a paused quiz session"""
        session = self.active_sessions.get(session_id)
        if not session or not session.pause_time:
            return False

        pause_duration = (datetime.now() - session.pause_time).total_seconds()
        session.pause_time = None

        # Adjust time limits if necessary
        if session.quiz_config.get('mode') in ['timed', 'exam']:
            # In exam mode, total time limit doesn't change
            pass

        return True

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the quiz session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {'error': 'Session not found'}

        attempt = session.attempt
        questions_answered = attempt.questions_answered
        total_questions = len(session.questions)

        # Calculate current performance
        if questions_answered > 0:
            accuracy = (attempt.correct_answers / questions_answered) * 100
            average_score = (attempt.total_points / questions_answered)
        else:
            accuracy = 0
            average_score = 0

        # Calculate time information
        if attempt.end_time:
            total_time = attempt.time_taken
        else:
            total_time = int((datetime.now() - attempt.start_time).total_seconds())

        return {
            'session_id': session_id,
            'user_id': session.user_id,
            'quiz_mode': session.quiz_config.get('mode', 'practice'),
            'questions_answered': questions_answered,
            'total_questions': total_questions,
            'correct_answers': attempt.correct_answers,
            'total_points': attempt.total_points,
            'max_points': attempt.max_points,
            'accuracy': accuracy,
            'average_score': average_score,
            'time_taken': total_time,
            'time_per_question': total_time / questions_answered if questions_answered > 0 else 0,
            'hints_used': attempt.hints_used,
            'completed': attempt.completed,
            'adaptive_difficulty': session.adaptive_difficulty if session.quiz_config.get('mode') == 'adaptive' else None
        }

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a quiz session and return final results"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {'error': 'Session not found'}

        # Complete the attempt if not already completed
        if not session.attempt.completed:
            session.attempt.end_time = datetime.now()
            session.attempt.completed = True
            session.attempt.time_taken = int((session.attempt.end_time - session.attempt.start_time).total_seconds())

        # Generate final results
        results = self.get_session_summary(session_id)

        # Add detailed answer breakdown
        answer_breakdown = []
        for i, question in enumerate(session.questions):
            if question.id in session.attempt.answers:
                answer_data = session.attempt.answers[question.id]
                answer_breakdown.append({
                    'question_number': i + 1,
                    'question_id': question.id,
                    'type': question.type.value,
                    'user_answer': answer_data['answer'],
                    'is_correct': answer_data['is_correct'],
                    'score': answer_data['score'],
                    'max_score': question.points,
                    'time_spent': answer_data.get('time_spent', 0),
                    'hints_used': answer_data.get('hints_used', 0)
                })
            else:
                answer_breakdown.append({
                    'question_number': i + 1,
                    'question_id': question.id,
                    'type': question.type.value,
                    'user_answer': None,
                    'is_correct': False,
                    'score': 0,
                    'max_score': question.points,
                    'time_spent': 0,
                    'hints_used': 0
                })

        results['answer_breakdown'] = answer_breakdown

        # Remove from active sessions
        del self.active_sessions[session_id]

        return results

    def _update_user_statistics(self, user_id: str, question: QuizQuestion,
                              is_correct: bool, time_spent: int):
        """Update user statistics based on question performance"""
        if user_id not in self.user_statistics:
            self.user_statistics[user_id] = {
                'total_questions_answered': 0,
                'total_correct': 0,
                'total_time_spent': 0,
                'weak_areas': {},
                'strong_areas': {},
                'recent_attempts': {},
                'difficulty_preference': 1,
                'question_type_performance': {}
            }

        stats = self.user_statistics[user_id]

        # Update basic statistics
        stats['total_questions_answered'] += 1
        if is_correct:
            stats['total_correct'] += 1
        stats['total_time_spent'] += time_spent

        # Update recent attempts (for spaced repetition)
        stats['recent_attempts'][question.id] = datetime.now()

        # Update weak/strong areas based on tags
        for tag in question.tags:
            if tag not in stats['weak_areas']:
                stats['weak_areas'][tag] = {'attempts': 0, 'correct': 0}
            if tag not in stats['strong_areas']:
                stats['strong_areas'][tag] = {'attempts': 0, 'correct': 0}

            stats['weak_areas'][tag]['attempts'] += 1
            stats['strong_areas'][tag]['attempts'] += 1

            if is_correct:
                stats['weak_areas'][tag]['correct'] += 1
                stats['strong_areas'][tag]['correct'] += 1

        # Calculate performance rates for weak areas
        for tag in stats['weak_areas']:
            attempts = stats['weak_areas'][tag]['attempts']
            correct = stats['weak_areas'][tag]['correct']
            if attempts > 0:
                performance_rate = correct / attempts
                # Higher values indicate weaker performance (areas needing more practice)
                stats['weak_areas'][tag]['performance_rate'] = 1 - performance_rate

        # Update question type performance
        qtype = question.type.value
        if qtype not in stats['question_type_performance']:
            stats['question_type_performance'][qtype] = {'attempts': 0, 'correct': 0}

        stats['question_type_performance'][qtype]['attempts'] += 1
        if is_correct:
            stats['question_type_performance'][qtype]['correct'] += 1

        self.save_data()

    def generate_review_quiz(self, user_id: str, num_questions: int = 20) -> str:
        """Generate a review quiz based on user's weak areas"""
        if user_id not in self.user_statistics:
            return self.start_quiz_session(user_id, {
                'mode': 'review',
                'num_questions': num_questions
            })

        stats = self.user_statistics[user_id]

        # Get questions from weak areas
        weak_questions = []
        for question in self.question_bank.values():
            for tag in question.tags:
                if tag in stats['weak_areas']:
                    performance_rate = stats['weak_areas'][tag].get('performance_rate', 0)
                    if performance_rate > 0.3:  # 30% or higher error rate
                        weak_questions.append((question, performance_rate))

        # Sort by performance rate (highest error rate first)
        weak_questions.sort(key=lambda x: x[1], reverse=True)

        # Select top questions
        selected_questions = [q for q, _ in weak_questions[:num_questions]]

        if not selected_questions:
            # Fallback to random questions
            selected_questions = random.sample(list(self.question_bank.values()), min(num_questions, len(self.question_bank)))

        # Create session
        session_id = f"review_{int(time.time() * 1000)}"
        session = QuizSession(
            session_id=session_id,
            user_id=user_id,
            quiz_config={'mode': 'review', 'num_questions': num_questions},
            questions=selected_questions,
            start_time=datetime.now()
        )

        session.attempt = QuizAttempt(
            attempt_id=f"attempt_{session_id}",
            user_id=user_id,
            quiz_id='review_quiz',
            start_time=datetime.now(),
            max_points=sum(q.points for q in selected_questions),
            mode=QuizMode.REVIEW
        )

        self.active_sessions[session_id] = session
        return session_id

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        if user_id not in self.user_statistics:
            return {'error': 'User not found'}

        stats = self.user_statistics[user_id]

        # Calculate derived statistics
        total_attempts = stats['total_questions_answered']
        total_correct = stats['total_correct']
        accuracy = (total_correct / total_attempts * 100) if total_attempts > 0 else 0
        avg_time_per_question = (stats['total_time_spent'] / total_attempts) if total_attempts > 0 else 0

        # Get top weak areas
        weak_areas_sorted = sorted(stats['weak_areas'].items(),
                                 key=lambda x: x[1].get('performance_rate', 0),
                                 reverse=True)

        # Get question type performance
        type_performance = {}
        for qtype, data in stats['question_type_performance'].items():
            attempts = data['attempts']
            correct = data['correct']
            type_performance[qtype] = {
                'accuracy': (correct / attempts * 100) if attempts > 0 else 0,
                'attempts': attempts
            }

        return {
            'user_id': user_id,
            'total_attempts': total_attempts,
            'total_correct': total_correct,
            'overall_accuracy': accuracy,
            'average_time_per_question': avg_time_per_question,
            'weak_areas': weak_areas_sorted[:5],  # Top 5 weak areas
            'question_type_performance': type_performance,
            'difficulty_preference': stats['difficulty_preference']
        }

    def _quiz_question_to_dict(self, question: QuizQuestion) -> Dict:
        """Convert QuizQuestion to dictionary for JSON serialization"""
        return {
            'id': question.id,
            'type': question.type.value,
            'content': question.content,
            'options': question.options,
            'correct_answer': question.correct_answer,
            'points': question.points,
            'time_limit': question.time_limit,
            'hints': question.hints,
            'explanation': question.explanation,
            'difficulty': question.difficulty,
            'section': question.section,
            'tags': question.tags
        }

    def _dict_to_quiz_question(self, data: Dict) -> QuizQuestion:
        """Convert dictionary to QuizQuestion object"""
        return QuizQuestion(
            id=data['id'],
            type=QuestionType(data['type']),
            content=data['content'],
            options=data.get('options'),
            correct_answer=data['correct_answer'],
            points=data.get('points', 10),
            time_limit=data.get('time_limit'),
            hints=data.get('hints', []),
            explanation=data.get('explanation', ''),
            difficulty=data.get('difficulty', 1),
            section=data.get('section', ''),
            tags=data.get('tags', [])
        )

if __name__ == "__main__":
    # Example usage
    quiz_system = InteractiveQuizSystem()

    # Create sample questions
    q1 = quiz_system.create_quiz_question(
        type="multiple_choice",
        content={
            "question": "What is the derivative of x²?",
            "explanation": "The power rule states that d/dx(x^n) = n*x^(n-1), so d/dx(x²) = 2x."
        },
        options=["2x", "x²", "2x²", "x"],
        correct_answer="2x",
        points=10,
        difficulty=1,
        section="calculus",
        tags=["derivatives", "power_rule", "calculus"]
    )

    q2 = quiz_system.create_quiz_question(
        type="true_false",
        content={
            "statement": "The dot product of two perpendicular vectors is zero.",
            "explanation": "Perpendicular vectors have a dot product of zero because cos(90°) = 0."
        },
        correct_answer=True,
        points=5,
        difficulty=1,
        section="linear_algebra",
        tags=["vectors", "dot_product", "geometry"]
    )

    q3 = quiz_system.create_quiz_question(
        type="fill_blank",
        content={
            "question": "In machine learning, ______ occurs when a model performs well on training data but poorly on new, unseen data.",
            "explanation": "Overfitting is when a model learns the training data too well, including noise and outliers."
        },
        correct_answer=["overfitting", "over-fitting"],
        points=10,
        difficulty=2,
        section="supervised_learning",
        tags=["overfitting", "generalization", "model_evaluation"]
    )

    print(f"Created {len(quiz_system.question_bank)} sample questions")

    # Start a quiz session
    quiz_config = {
        'mode': 'practice',
        'section': 'calculus',
        'difficulty': 1,
        'num_questions': 5,
        'quiz_id': 'sample_quiz'
    }

    session_id = quiz_system.start_quiz_session('user_001', quiz_config)
    print(f"Started quiz session: {session_id}")

    # Get current question
    current_question = quiz_system.get_current_question(session_id)
    if current_question:
        print(f"Current question: {current_question['content']['question']}")

    # Submit an answer
    result = quiz_system.submit_answer(session_id, q1.id, "2x", 30)
    print(f"Answer result: {result}")

    # Get session summary
    summary = quiz_system.get_session_summary(session_id)
    print(f"Session summary: {summary}")

    # End session
    final_results = quiz_system.end_session(session_id)
    print(f"Final results: {final_results}")

    print("Interactive quiz system initialized successfully!")