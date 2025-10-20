#!/usr/bin/env python3
"""
Comprehensive Interactive Learning Management System
Enhances the existing assessment framework with advanced interactive features
"""

import json
import os
import sys
import uuid
import hashlib
import random
import time
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Import existing assessment components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'assessment'))
from assessment_engine_core import AssessmentEngine, QuestionType, DifficultyLevel, AssessmentType
from progress_tracking_analytics import ProgressTracker, PerformanceAnalytics, LearningPath
from interactive_quiz_system import InteractiveQuizSystem, QuizMode
from achievement_certification_system import AchievementSystem, AchievementType, CertificateType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Different learning modes available in the system"""
    GUIDED = "guided"           # Step-by-step guidance
    EXPLORATORY = "exploratory" # Free exploration
    CHALLENGE = "challenge"     # Timed challenges
    COLLABORATIVE = "collaborative" # Group learning
    REMEDIATION = "remediation" # Targeted gap filling
    ADVANCED = "advanced"       # Research-level learning

class ContentType(Enum):
    """Types of learning content"""
    INTERACTIVE_TUTORIAL = "interactive_tutorial"
    HANDS_ON_EXERCISE = "hands_on_exercise"
    CODING_CHALLENGE = "coding_challenge"
    CASE_STUDY = "case_study"
    SIMULATION = "simulation"
    GAMIFIED_MISSION = "gamified_mission"
    PEER_REVIEW = "peer_review"
    MENTORSHIP_SESSION = "mentorship_session"

class SkillLevel(Enum):
    """Skill levels for progression tracking"""
    NOVICE = 1
    BEGINNER = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5
    MASTER = 6
    INNOVATOR = 7

@dataclass
class LearningProfile:
    """Comprehensive user learning profile"""
    user_id: str
    display_name: str
    skill_level: SkillLevel
    preferred_learning_mode: LearningMode
    learning_style: Dict[str, float]  # visual, auditory, kinesthetic, reading
    cognitive_load_tolerance: float  # 0.0 - 1.0
    attention_span_minutes: int
    mastery_pace: str  # slow, moderate, fast, adaptive
    goal_orientation: str  # performance, learning, social, achievement
    collaboration_preference: float  # 0.0 - 1.0
    last_active: datetime
    total_learning_hours: float
    streak_days: int
    current_mood: str  # motivated, tired, frustrated, curious
    personal_interests: List[str]
    career_goals: List[str]
    preferred_time_of_day: str  # morning, afternoon, evening, night

@dataclass
class InteractiveExercise:
    """Interactive exercise definition"""
    exercise_id: str
    title: str
    description: str
    content_type: ContentType
    section_id: str
    difficulty: DifficultyLevel
    estimated_minutes: int
    prerequisites: List[str]
    learning_objectives: List[str]
    interactive_elements: List[Dict[str, Any]]
    real_time_feedback: bool
    adaptive_hints: bool
    live_code_execution: bool
    peer_interaction_allowed: bool
    gamification_elements: Dict[str, Any]

@dataclass
class LearningSession:
    """Active learning session data"""
    session_id: str
    user_id: str
    start_time: datetime
    current_exercise: Optional[str]
    exercises_completed: List[str]
    time_spent_minutes: int
    engagement_score: float
    frustration_indicators: List[str]
    achievement_milestones: List[str]
    collaboration_events: List[Dict[str, Any]]
    real_time_data: Dict[str, Any]

@dataclass
class LearningPathNode:
    """Individual node in a learning path"""
    node_id: str
    content_id: str
    content_type: ContentType
    prerequisites: List[str]
    mastery_threshold: float
    estimated_time: int
    adaptive_adjustments: Dict[str, Any]
    personalization_weights: Dict[str, float]

class SpacedRepetitionScheduler:
    """Advanced spaced repetition system for optimal learning retention"""

    def __init__(self):
        self.forgetting_curve_params = {
            'initial_strength': 0.9,
            'decay_rate': 0.5,
            'stability_multiplier': 2.5,
            'difficulty_factor': 1.3
        }
        self.review_intervals = {
            'easy': [1, 3, 7, 14, 30, 60, 120],
            'medium': [1, 2, 5, 10, 20, 40, 80],
            'hard': [1, 1, 3, 6, 12, 24, 48]
        }

    def calculate_next_review(self,
                            mastery_level: float,
                            difficulty_rating: int,  # 1-4 scale
                            days_since_last_review: int,
                            previous_interval: int,
                            performance_score: float) -> datetime:
        """Calculate optimal next review time using spaced repetition algorithm"""

        # Calculate retention probability
        retention = self._calculate_retention_probability(
            mastery_level, days_since_last_review, previous_interval
        )

        # Adjust interval based on performance
        if performance_score >= 0.8:
            multiplier = self.forgetting_curve_params['stability_multiplier']
        elif performance_score >= 0.6:
            multiplier = 1.5
        else:
            multiplier = 0.5

        # Apply difficulty factor
        difficulty_multiplier = {
            1: 2.0,  # Easy
            2: 1.5,  # Medium
            3: 1.0,  # Hard
            4: 0.5   # Very Hard
        }.get(difficulty_rating, 1.0)

        # Calculate new interval
        if previous_interval == 0:
            new_interval = 1
        else:
            new_interval = max(1, int(previous_interval * multiplier * difficulty_multiplier))

        # Cap maximum interval at 180 days
        new_interval = min(new_interval, 180)

        next_review = datetime.now(timezone.utc) + timedelta(days=new_interval)
        return next_review

    def _calculate_retention_probability(self,
                                      mastery_level: float,
                                      days_elapsed: int,
                                      previous_interval: int) -> float:
        """Calculate probability of retention using forgetting curve"""
        if previous_interval == 0:
            return 1.0

        # Modified forgetting curve equation
        strength = self.forgetting_curve_params['initial_strength'] * mastery_level
        decay_rate = self.forgetting_curve_params['decay_rate']

        retention = strength * np.exp(-decay_rate * days_elapsed / previous_interval)
        return max(0.1, min(1.0, retention))

class AdaptiveLearningEngine:
    """Core adaptive learning engine for personalized education"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        # Learning analytics models
        self.skill_decay_model = self._load_skill_decay_model()
        self.engagement_predictor = self._load_engagement_predictor()
        self.difficulty_adjuster = self._load_difficulty_adjuster()

        # User learning data
        self.user_learning_patterns = {}
        self.content_effectiveness = {}
        self.learning_optimization_params = {}

        # Spaced repetition scheduler
        self.spaced_repetition = SpacedRepetitionScheduler()

    def _load_skill_decay_model(self) -> Dict[str, Any]:
        """Load or initialize skill decay prediction model"""
        model_file = self.data_path / "skill_decay_model.pkl"
        if model_file.exists():
            with open(model_file, 'rb') as f:
                return pickle.load(f)

        # Default decay model parameters
        return {
            'decay_rate': 0.1,  # Daily decay rate
            'practice_effect': 0.8,  # Effectiveness of practice
            'complexity_factor': 1.2,  # How complexity affects decay
            'mastery_plateau': 0.9  # Maximum mastery level
        }

    def _load_engagement_predictor(self) -> Dict[str, Any]:
        """Load or initialize engagement prediction model"""
        model_file = self.data_path / "engagement_predictor.pkl"
        if model_file.exists():
            with open(model_file, 'rb') as f:
                return pickle.load(f)

        return {
            'attention_threshold': 20,  # Minutes
            'difficulty_threshold': 0.7,  # Optimal difficulty
            'frustration_indicators': [
                'time_per_question_increase',
                'hint_usage_spike',
                'error_rate_increase',
                'session_abandonment'
            ],
            'engagement_boosters': [
                'immediate_feedback',
                'gamification_elements',
                'social_comparison',
                'progress_visualization'
            ]
        }

    def _load_difficulty_adjuster(self) -> Dict[str, Any]:
        """Load or initialize difficulty adjustment model"""
        model_file = self.data_path / "difficulty_adjuster.pkl"
        if model_file.exists():
            with open(model_file, 'rb') as f:
                return pickle.load(f)

        return {
            'target_success_rate': 0.7,  # Optimal success rate for learning
            'adjustment_factor': 0.1,  # How much to adjust difficulty
            'min_difficulty': 1,
            'max_difficulty': 10,
            'performance_window': 5,  # Number of recent performances to consider
            'difficulty_decay_rate': 0.05  # Gradual return to baseline
        }

    def calculate_optimal_difficulty(self,
                                   user_id: str,
                                   section_id: str,
                                   recent_performances: List[float]) -> DifficultyLevel:
        """Calculate optimal difficulty level for next exercise"""

        if not recent_performances:
            return DifficultyLevel.BEGINNER

        avg_performance = np.mean(recent_performances[-5:])  # Last 5 performances
        target_rate = self.difficulty_adjuster['target_success_rate']

        # Get current difficulty
        current_difficulty = self._get_user_current_difficulty(user_id, section_id)

        # Adjust based on performance
        if avg_performance > target_rate + 0.1:
            # Increase difficulty
            new_difficulty = min(10, current_difficulty + 1)
        elif avg_performance < target_rate - 0.1:
            # Decrease difficulty
            new_difficulty = max(1, current_difficulty - 1)
        else:
            # Maintain current difficulty
            new_difficulty = current_difficulty

        # Map to DifficultyLevel enum
        if new_difficulty <= 2:
            return DifficultyLevel.BEGINNER
        elif new_difficulty <= 4:
            return DifficultyLevel.INTERMEDIATE
        elif new_difficulty <= 6:
            return DifficultyLevel.ADVANCED
        elif new_difficulty <= 8:
            return DifficultyLevel.EXPERT
        else:
            return DifficultyLevel.MASTER

    def generate_personalized_learning_path(self,
                                          user_id: str,
                                          goal_section: str,
                                          current_skills: Dict[str, float],
                                          time_constraint_hours: int,
                                          preferred_content_types: List[ContentType]) -> List[LearningPathNode]:
        """Generate personalized learning path based on user profile and constraints"""

        # Get all available content for the target section
        available_content = self._get_section_content(goal_section)

        # Filter by preferred content types
        if preferred_content_types:
            available_content = [c for c in available_content
                               if ContentType(c['type']) in preferred_content_types]

        # Sort by prerequisites and difficulty
        content_graph = self._build_content_dependency_graph(available_content)
        topological_order = self._topological_sort(content_graph)

        # Generate path nodes
        path_nodes = []
        total_time = 0

        for content_id in topological_order:
            if total_time >= time_constraint_hours * 60:  # Convert to minutes
                break

            content_data = self._get_content_data(content_id)

            # Skip if prerequisites not met
            if not self._prerequisites_met(content_data, current_skills):
                continue

            # Calculate personalization weights
            personalization_weights = self._calculate_personalization_weights(
                user_id, content_data, current_skills
            )

            node = LearningPathNode(
                node_id=str(uuid.uuid4()),
                content_id=content_id,
                content_type=ContentType(content_data['type']),
                prerequisites=content_data.get('prerequisites', []),
                mastery_threshold=self._calculate_mastery_threshold(content_data, current_skills),
                estimated_time=content_data.get('estimated_minutes', 30),
                adaptive_adjustments={},
                personalization_weights=personalization_weights
            )

            path_nodes.append(node)
            total_time += node.estimated_time

        return path_nodes

    def predict_learning_outcome(self,
                               user_id: str,
                               learning_path: List[LearningPathNode],
                               time_constraint_hours: int) -> Dict[str, float]:
        """Predict learning outcomes for a given path"""

        user_profile = self._get_user_learning_profile(user_id)

        # Calculate predicted mastery levels
        predicted_mastery = {}
        current_skills = self._get_user_current_skills(user_id)

        for node in learning_path:
            content_data = self._get_content_data(node.content_id)
            skill_gains = self._predict_skill_gains(user_profile, content_data, current_skills)

            for skill, gain in skill_gains.items():
                if skill not in predicted_mastery:
                    predicted_mastery[skill] = current_skills.get(skill, 0.0)
                predicted_mastery[skill] = min(1.0, predicted_mastery[skill] + gain)

            current_skills = predicted_mastery.copy()

        # Calculate engagement probability
        engagement_score = self._predict_engagement_score(user_profile, learning_path)

        # Calculate completion probability
        total_time = sum(node.estimated_time for node in learning_path)
        completion_probability = min(1.0, (time_constraint_hours * 60) / total_time)

        return {
            'average_mastery_gain': np.mean(list(predicted_mastery.values())) -
                                  np.mean(list(self._get_user_current_skills(user_id).values())),
            'predicted_mastery_levels': predicted_mastery,
            'engagement_probability': engagement_score,
            'completion_probability': completion_probability,
            'time_efficiency': total_time / 60,  # Hours
            'learning_efficiency': np.mean(list(predicted_mastery.values())) / (total_time / 60)
        }

class InteractiveLearningManagementSystem:
    """Main interactive learning management system"""

    def __init__(self, data_path: str = "learning_system_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        # Initialize components
        self.adaptive_engine = AdaptiveLearningEngine(self.data_path / "adaptive")
        self.assessment_engine = AssessmentEngine(self.data_path / "assessment")
        self.progress_tracker = ProgressTracker(self.data_path / "progress")
        self.quiz_system = InteractiveQuizSystem(self.data_path / "quiz")
        self.achievement_system = AchievementSystem(self.data_path / "achievements")

        # Learning data storage
        self.learning_profiles = {}
        self.active_sessions = {}
        self.exercise_library = {}
        self.learning_paths = {}

        # Real-time data processing
        self.real_time_queue = queue.Queue()
        self.analytics_executor = ThreadPoolExecutor(max_workers=4)

        # Load existing data
        self._load_learning_data()

        logger.info("Interactive Learning Management System initialized")

    def _load_learning_data(self):
        """Load existing learning data"""
        # Load learning profiles
        profiles_file = self.data_path / "learning_profiles.json"
        if profiles_file.exists():
            with open(profiles_file, 'r') as f:
                profiles_data = json.load(f)
                for user_id, profile_data in profiles_data.items():
                    profile_data['skill_level'] = SkillLevel(profile_data['skill_level'])
                    profile_data['preferred_learning_mode'] = LearningMode(profile_data['preferred_learning_mode'])
                    profile_data['last_active'] = datetime.fromisoformat(profile_data['last_active'])
                    self.learning_profiles[user_id] = LearningProfile(**profile_data)

        # Load exercise library
        exercises_file = self.data_path / "exercise_library.json"
        if exercises_file.exists():
            with open(exercises_file, 'r') as f:
                exercises_data = json.load(f)
                for ex_id, ex_data in exercises_data.items():
                    ex_data['content_type'] = ContentType(ex_data['content_type'])
                    ex_data['difficulty'] = DifficultyLevel(ex_data['difficulty'])
                    self.exercise_library[ex_id] = InteractiveExercise(**ex_data)

        # Load active sessions
        sessions_file = self.data_path / "active_sessions.json"
        if sessions_file.exists():
            with open(sessions_file, 'r') as f:
                sessions_data = json.load(f)
                for session_id, session_data in sessions_data.items():
                    session_data['start_time'] = datetime.fromisoformat(session_data['start_time'])
                    self.active_sessions[session_id] = LearningSession(**session_data)

    def create_learning_profile(self,
                             user_id: str,
                             display_name: str,
                             initial_assessment_results: Optional[Dict[str, Any]] = None,
                             preferences: Optional[Dict[str, Any]] = None) -> LearningProfile:
        """Create comprehensive learning profile for user"""

        # Determine initial skill level
        if initial_assessment_results:
            skill_level = self._determine_skill_level(initial_assessment_results)
        else:
            skill_level = SkillLevel.BEGINNER

        # Set preferences or use defaults
        if preferences:
            learning_mode = LearningMode(preferences.get('learning_mode', 'guided'))
            learning_style = preferences.get('learning_style', {
                'visual': 0.4, 'auditory': 0.2, 'kinesthetic': 0.3, 'reading': 0.1
            })
            cognitive_load = preferences.get('cognitive_load_tolerance', 0.7)
            attention_span = preferences.get('attention_span_minutes', 45)
            pace = preferences.get('mastery_pace', 'moderate')
            goal_orientation = preferences.get('goal_orientation', 'learning')
            collaboration_pref = preferences.get('collaboration_preference', 0.5)
        else:
            learning_mode = LearningMode.GUIDED
            learning_style = {'visual': 0.4, 'auditory': 0.2, 'kinesthetic': 0.3, 'reading': 0.1}
            cognitive_load = 0.7
            attention_span = 45
            pace = 'moderate'
            goal_orientation = 'learning'
            collaboration_pref = 0.5

        profile = LearningProfile(
            user_id=user_id,
            display_name=display_name,
            skill_level=skill_level,
            preferred_learning_mode=learning_mode,
            learning_style=learning_style,
            cognitive_load_tolerance=cognitive_load,
            attention_span_minutes=attention_span,
            mastery_pace=pace,
            goal_orientation=goal_orientation,
            collaboration_preference=collaboration_pref,
            last_active=datetime.now(timezone.utc),
            total_learning_hours=0.0,
            streak_days=0,
            current_mood='motivated',
            personal_interests=preferences.get('personal_interests', []) if preferences else [],
            career_goals=preferences.get('career_goals', []) if preferences else [],
            preferred_time_of_day=preferences.get('preferred_time_of_day', 'morning') if preferences else 'morning'
        )

        self.learning_profiles[user_id] = profile
        self._save_learning_profiles()

        # Generate initial learning path
        self._generate_initial_learning_path(user_id)

        logger.info(f"Created learning profile for user: {user_id}")
        return profile

    def start_learning_session(self,
                             user_id: str,
                             learning_mode: LearningMode,
                             section_id: Optional[str] = None,
                             time_constraint_minutes: Optional[int] = None) -> str:
        """Start a new learning session"""

        if user_id not in self.learning_profiles:
            raise ValueError(f"User profile not found: {user_id}")

        # Create session
        session_id = str(uuid.uuid4())
        session = LearningSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(timezone.utc),
            current_exercise=None,
            exercises_completed=[],
            time_spent_minutes=0,
            engagement_score=0.0,
            frustration_indicators=[],
            achievement_milestones=[],
            collaboration_events=[],
            real_time_data={}
        )

        self.active_sessions[session_id] = session

        # Update user profile
        self.learning_profiles[user_id].last_active = datetime.now(timezone.utc)

        # Get or generate appropriate exercise
        if section_id:
            exercise = self._get_next_exercise(user_id, section_id, learning_mode)
            if exercise:
                session.current_exercise = exercise.exercise_id

        self._save_active_sessions()

        logger.info(f"Started learning session: {session_id} for user: {user_id}")
        return session_id

    def get_recommended_exercise(self,
                               session_id: str,
                               user_preferences: Optional[Dict[str, Any]] = None) -> Optional[InteractiveExercise]:
        """Get next recommended exercise based on adaptive learning"""

        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.active_sessions[session_id]
        user_id = session.user_id
        profile = self.learning_profiles[user_id]

        # Get user's current performance data
        recent_performances = self._get_recent_performances(user_id)
        current_skills = self._get_user_current_skills(user_id)

        # Determine optimal difficulty
        current_section = self._get_current_learning_section(user_id)
        optimal_difficulty = self.adaptive_engine.calculate_optimal_difficulty(
            user_id, current_section, recent_performances
        )

        # Get exercises matching criteria
        suitable_exercises = self._find_suitable_exercises(
            current_section,
            optimal_difficulty,
            profile.preferred_learning_mode,
            profile.learning_style,
            user_preferences
        )

        if not suitable_exercises:
            return None

        # Select best exercise using adaptive scoring
        best_exercise = self._select_best_exercise(
            suitable_exercises,
            user_id,
            current_skills,
            profile
        )

        # Update session
        session.current_exercise = best_exercise.exercise_id

        self._save_active_sessions()

        return best_exercise

    def submit_exercise_attempt(self,
                              session_id: str,
                              exercise_id: str,
                              answers: Dict[str, Any],
                              time_spent_seconds: int,
                              feedback_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process exercise submission and provide adaptive feedback"""

        if session_id not in self.active_sessions:
            raise ValueError(f"Session not found: {session_id}")

        session = self.active_sessions[session_id]
        user_id = session.user_id

        # Get exercise details
        exercise = self.exercise_library.get(exercise_id)
        if not exercise:
            raise ValueError(f"Exercise not found: {exercise_id}")

        # Evaluate submission
        evaluation_result = self._evaluate_exercise_submission(
            exercise, answers, user_id
        )

        # Update session data
        session.exercises_completed.append(exercise_id)
        session.time_spent_minutes += time_spent_seconds // 60

        # Process real-time data
        if feedback_data:
            self._process_real_time_feedback(session_id, feedback_data)

        # Update learning analytics
        self._update_learning_analytics(user_id, exercise_id, evaluation_result)

        # Check for achievements
        new_achievements = self._check_session_achievements(session, evaluation_result)
        session.achievement_milestones.extend(new_achievements)

        # Schedule spaced repetition if needed
        if evaluation_result['mastery_level'] < 0.8:
            self._schedule_review_session(user_id, exercise_id, evaluation_result)

        # Generate adaptive feedback
        adaptive_feedback = self._generate_adaptive_feedback(
            user_id, exercise, evaluation_result, session
        )

        # Save session data
        self._save_active_sessions()

        result = {
            'evaluation_result': evaluation_result,
            'adaptive_feedback': adaptive_feedback,
            'new_achievements': new_achievements,
            'next_recommendations': self._generate_next_recommendations(user_id, session),
            'learning_insights': self._generate_learning_insights(user_id, exercise_id)
        }

        return result

    def get_comprehensive_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive learning analytics for a user"""

        if user_id not in self.learning_profiles:
            raise ValueError(f"User profile not found: {user_id}")

        profile = self.learning_profiles[user_id]

        # Get basic analytics from progress tracker
        basic_analytics = self.progress_tracker.get_comprehensive_analytics(user_id)

        # Get adaptive learning insights
        adaptive_insights = self._get_adaptive_learning_insights(user_id)

        # Get engagement metrics
        engagement_metrics = self._calculate_engagement_metrics(user_id)

        # Get skill development trajectory
        skill_trajectory = self._calculate_skill_trajectory(user_id)

        # Get learning efficiency metrics
        efficiency_metrics = self._calculate_learning_efficiency(user_id)

        # Get collaboration insights
        collaboration_insights = self._get_collaboration_insights(user_id)

        # Get predictive analytics
        predictive_analytics = self._generate_predictive_analytics(user_id)

        # Get gamification metrics
        gamification_metrics = self._get_gamification_metrics(user_id)

        return {
            'user_profile': asdict(profile),
            'basic_analytics': basic_analytics,
            'adaptive_insights': adaptive_insights,
            'engagement_metrics': engagement_metrics,
            'skill_trajectory': skill_trajectory,
            'efficiency_metrics': efficiency_metrics,
            'collaboration_insights': collaboration_insights,
            'predictive_analytics': predictive_analytics,
            'gamification_metrics': gamification_metrics,
            'recommendations': self._generate_comprehensive_recommendations(user_id),
            'next_milestones': self._get_upcoming_milestones(user_id)
        }

    def get_leaderboard(self,
                       category: str = "overall",
                       time_period: str = "week",
                       limit: int = 50) -> List[Dict[str, Any]]:
        """Get leaderboard data for gamification"""

        # Get all users
        all_users = list(self.learning_profiles.keys())

        leaderboard_data = []

        for user_id in all_users:
            profile = self.learning_profiles[user_id]

            # Calculate score based on category
            if category == "overall":
                score = self._calculate_overall_score(user_id)
            elif category == "skill_mastery":
                score = self._calculate_skill_mastery_score(user_id)
            elif category == "learning_streak":
                score = profile.streak_days
            elif category == "engagement":
                score = self._calculate_engagement_score(user_id)
            elif category == "collaboration":
                score = self._calculate_collaboration_score(user_id)
            else:
                score = 0

            leaderboard_data.append({
                'user_id': user_id,
                'display_name': profile.display_name,
                'score': score,
                'skill_level': profile.skill_level.name,
                'avatar_url': f"/api/users/{user_id}/avatar",
                'badges_count': len(self.achievement_system.get_user_achievements(user_id)),
                'certificates_count': len(self.achievement_system.get_user_certificates(user_id))
            })

        # Sort by score
        leaderboard_data.sort(key=lambda x: x['score'], reverse=True)

        # Add ranks
        for i, entry in enumerate(leaderboard_data[:limit]):
            entry['rank'] = i + 1

        return leaderboard_data[:limit]

    def _determine_skill_level(self, assessment_results: Dict[str, Any]) -> SkillLevel:
        """Determine user's skill level from assessment results"""

        overall_score = assessment_results.get('overall_score', 0)

        if overall_score >= 0.95:
            return SkillLevel.INNOVATOR
        elif overall_score >= 0.85:
            return SkillLevel.MASTER
        elif overall_score >= 0.75:
            return SkillLevel.EXPERT
        elif overall_score >= 0.60:
            return SkillLevel.ADVANCED
        elif overall_score >= 0.40:
            return SkillLevel.INTERMEDIATE
        elif overall_score >= 0.20:
            return SkillLevel.BEGINNER
        else:
            return SkillLevel.NOVICE

    def _get_next_exercise(self,
                         user_id: str,
                         section_id: str,
                         learning_mode: LearningMode) -> Optional[InteractiveExercise]:
        """Get next exercise for user based on learning mode"""

        profile = self.learning_profiles[user_id]

        # Filter exercises by section and learning mode
        suitable_exercises = [
            ex for ex in self.exercise_library.values()
            if ex.section_id == section_id and
            self._is_suitable_for_learning_mode(ex, learning_mode)
        ]

        if not suitable_exercises:
            return None

        # Sort by difficulty and user skill level
        current_difficulty = self._get_user_current_difficulty(user_id, section_id)
        suitable_exercises.sort(key=lambda x: abs(x.difficulty.value - current_difficulty))

        return suitable_exercises[0]

    def _is_suitable_for_learning_mode(self,
                                     exercise: InteractiveExercise,
                                     learning_mode: LearningMode) -> bool:
        """Check if exercise is suitable for learning mode"""

        mode_compatibility = {
            LearningMode.GUIDED: [ContentType.INTERACTIVE_TUTORIAL, ContentType.HANDS_ON_EXERCISE],
            LearningMode.EXPLORATORY: [ContentType.SIMULATION, ContentType.CASE_STUDY],
            LearningMode.CHALLENGE: [ContentType.CODING_CHALLENGE, ContentType.GAMIFIED_MISSION],
            LearningMode.COLLABORATIVE: [ContentType.PEER_REVIEW],
            LearningMode.REMEDIATION: [ContentType.INTERACTIVE_TUTORIAL, ContentType.HANDS_ON_EXERCISE],
            LearningMode.ADVANCED: [ContentType.CODING_CHALLENGE, ContentType.CASE_STUDY]
        }

        return exercise.content_type in mode_compatibility.get(learning_mode, [])

    def _evaluate_exercise_submission(self,
                                    exercise: InteractiveExercise,
                                    answers: Dict[str, Any],
                                    user_id: str) -> Dict[str, Any]:
        """Evaluate exercise submission with detailed feedback"""

        # Use assessment engine for evaluation
        evaluation_result = {
            'score': 0.0,
            'mastery_level': 0.0,
            'correct_answers': 0,
            'total_questions': 0,
            'detailed_feedback': {},
            'strengths': [],
            'weaknesses': [],
            'time_analysis': {},
            'improvement_suggestions': []
        }

        # Process based on content type
        if exercise.content_type == ContentType.CODING_CHALLENGE:
            evaluation_result.update(self._evaluate_coding_challenge(exercise, answers, user_id))
        elif exercise.content_type == ContentType.INTERACTIVE_TUTORIAL:
            evaluation_result.update(self._evaluate_tutorial_exercise(exercise, answers, user_id))
        elif exercise.content_type == ContentType.CASE_STUDY:
            evaluation_result.update(self._evaluate_case_study(exercise, answers, user_id))
        else:
            # Generic evaluation
            evaluation_result.update(self._evaluate_generic_exercise(exercise, answers, user_id))

        return evaluation_result

    def _save_learning_profiles(self):
        """Save learning profiles to file"""
        profiles_file = self.data_path / "learning_profiles.json"
        profiles_data = {}

        for user_id, profile in self.learning_profiles.items():
            profile_dict = asdict(profile)
            profile_dict['skill_level'] = profile.skill_level.value
            profile_dict['preferred_learning_mode'] = profile.preferred_learning_mode.value
            profile_dict['last_active'] = profile.last_active.isoformat()
            profiles_data[user_id] = profile_dict

        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=2)

    def _save_active_sessions(self):
        """Save active sessions to file"""
        sessions_file = self.data_path / "active_sessions.json"
        sessions_data = {}

        for session_id, session in self.active_sessions.items():
            session_dict = asdict(session)
            session_dict['start_time'] = session.start_time.isoformat()
            sessions_data[session_id] = session_dict

        with open(sessions_file, 'w') as f:
            json.dump(sessions_data, f, indent=2)

    # Additional helper methods would be implemented here...
    # (Due to space constraints, I'm showing the main structure)

# Main execution
if __name__ == "__main__":
    # Initialize the learning management system
    lms = InteractiveLearningManagementSystem()

    # Example usage
    print("Interactive Learning Management System initialized")
    print(f"Learning profiles loaded: {len(lms.learning_profiles)}")
    print(f"Exercises available: {len(lms.exercise_library)}")
    print(f"Active sessions: {len(lms.active_sessions)}")