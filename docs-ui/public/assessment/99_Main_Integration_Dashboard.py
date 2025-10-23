#!/usr/bin/env python3
"""
Main Assessment System Integration and Dashboard
Combines all assessment components into a cohesive system
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Import all assessment components
from assessment_engine_core import AssessmentEngine, QuestionType, DifficultyLevel, AssessmentType
from progress_tracking_analytics import ProgressTracker, PerformanceAnalytics, LearningPath
from interactive_quiz_system import InteractiveQuizSystem, QuizMode
from achievement_certification_system import AchievementSystem, AchievementType, CertificateType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssessmentSystemDashboard:
    """Main dashboard integrating all assessment components"""

    def __init__(self, data_path: str = "assessment_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        # Initialize all components
        self.assessment_engine = AssessmentEngine(data_path)
        self.progress_tracker = ProgressTracker(data_path)
        self.quiz_system = InteractiveQuizSystem(data_path)
        self.achievement_system = AchievementSystem(data_path)

        # System configuration
        self.config = self._load_config()

        # Active sessions management
        self.active_user_sessions = {}

        logger.info("Assessment System Dashboard initialized successfully")

    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        config_file = self.data_path / "config.json"
        default_config = {
            "default_difficulty": 2,
            "session_timeout_minutes": 60,
            "max_questions_per_assessment": 50,
            "enable_real_time_analytics": True,
            "leaderboard_refresh_minutes": 30,
            "certificate_auto_issue": True,
            "achievement_notifications": True,
            "adaptive_learning_enabled": True,
            "data_retention_days": 365
        }

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")

        return default_config

    def save_config(self):
        """Save system configuration"""
        config_file = self.data_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def user_login(self, user_id: str, user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle user login and session initialization"""
        # Initialize user profile if not exists
        if user_id not in self.progress_tracker.user_data:
            if user_profile is None:
                user_profile = {
                    "display_name": user_id,
                    "experience_level": "beginner",
                    "learning_goals": [],
                    "preferred_sections": []
                }

            self.progress_tracker._initialize_user_profile(user_id)
            user_profile_data = self.progress_tracker.user_data[user_id]
            user_profile_data.update(user_profile)
            self.progress_tracker.save_data()

        # Create user session
        session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
        self.active_user_sessions[session_id] = {
            "user_id": user_id,
            "login_time": datetime.now(),
            "last_activity": datetime.now(),
            "session_data": {}
        }

        # Get user data for achievement checking
        user_data = self._prepare_user_data_for_achievements(user_id)

        # Check for new achievements and certificates
        new_achievements = self.achievement_system.check_achievements(user_id, user_data)
        new_certificates = self.achievement_system.check_certificates(user_id, user_data)

        return {
            "session_id": session_id,
            "user_id": user_id,
            "new_achievements": len(new_achievements),
            "new_certificates": len(new_certificates),
            "welcome_message": self._generate_welcome_message(user_id)
        }

    def _prepare_user_data_for_achievements(self, user_id: str) -> Dict[str, Any]:
        """Prepare user data for achievement checking"""
        user_profile = self.progress_tracker.user_data.get(user_id, {})

        # Calculate various metrics
        assessment_history = user_profile.get('assessment_history', [])
        completed_assessments = len(assessment_history)

        if completed_assessments > 0:
            average_score = sum(a['score'] for a in assessment_history) / completed_assessments
            perfect_scores = sum(1 for a in assessment_history if a['score'] == 100)
        else:
            average_score = 0
            perfect_scores = 0

        # Calculate consecutive days
        consecutive_days = self._calculate_consecutive_days(user_id)

        # Get completed sections
        completed_sections = set()
        for assessment in assessment_history:
            section_id = assessment.get('section_id', '')
            if assessment['score'] >= 70:  # 70% threshold for section completion
                completed_sections.add(section_id)

        # Get mastered skills
        mastered_skills = []
        skills = user_profile.get('skills', {})
        for skill_id, skill_data in skills.items():
            if skill_data['level'] >= 90:  # 90% threshold for mastery
                mastered_skills.append(skill_id)

        return {
            "user_id": user_id,
            "assessments_completed": completed_assessments,
            "average_score": average_score,
            "consecutive_days": consecutive_days,
            "perfect_scores": perfect_scores,
            "completed_sections": list(completed_sections),
            "total_assessments": completed_assessments,
            "mastered_skills": mastered_skills,
            "practical_projects": user_profile.get('practical_projects', 0),
            "capstone_completed": user_profile.get('capstone_completed', False)
        }

    def _calculate_consecutive_days(self, user_id: str) -> int:
        """Calculate consecutive days of activity"""
        user_profile = self.progress_tracker.user_data.get(user_id, {})
        assessment_history = user_profile.get('assessment_history', [])

        if not assessment_history:
            return 0

        # Get dates of assessments
        dates = [datetime.fromisoformat(a['timestamp']).date() for a in assessment_history]
        dates = sorted(set(dates))  # Remove duplicates and sort

        # Calculate consecutive days from today
        today = datetime.now().date()
        consecutive_days = 0

        for i in range(len(dates)):
            check_date = today - timedelta(days=i)
            if check_date in dates:
                consecutive_days += 1
            else:
                break

        return consecutive_days

    def _generate_welcome_message(self, user_id: str) -> str:
        """Generate personalized welcome message"""
        user_data = self.progress_tracker.user_data.get(user_id, {})
        current_level = user_data.get('current_level', 1)
        experience_points = user_data.get('experience_points', 0)

        # Determine experience level
        if current_level >= 20:
            level_title = "Expert"
        elif current_level >= 10:
            level_title = "Advanced"
        elif current_level >= 5:
            level_title = "Intermediate"
        else:
            level_title = "Beginner"

        return f"Welcome back! You're a {level_title} learner (Level {current_level}) with {experience_points} experience points."

    def start_assessment(self, user_id: str, assessment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new assessment session"""
        # Validate user session
        session_id = self._get_active_session(user_id)
        if not session_id:
            raise ValueError("No active user session found")

        # Determine assessment type and generate accordingly
        assessment_type = assessment_config.get('type', 'quiz')

        if assessment_type == 'quiz':
            # Use quiz system
            quiz_session_id = self.quiz_system.start_quiz_session(user_id, assessment_config)
            return {
                "assessment_id": quiz_session_id,
                "type": "quiz",
                "session_id": session_id,
                "total_questions": len(self.quiz_system.active_sessions[quiz_session_id].questions),
                "time_limit": assessment_config.get('time_limit'),
                "mode": assessment_config.get('mode', 'practice')
            }

        elif assessment_type == 'adaptive':
            # Generate adaptive assessment using assessment engine
            section_id = assessment_config.get('section_id', '01_Foundational_Machine_Learning')
            difficulty = DifficultyLevel(assessment_config.get('difficulty', 2))
            num_questions = assessment_config.get('num_questions', 20)

            assessment = self.assessment_engine.generate_assessment(
                section_id=section_id,
                difficulty=difficulty,
                num_questions=num_questions,
                assessment_type=AssessmentType.ADAPTIVE
            )

            # Store assessment in user session
            self.active_user_sessions[session_id]['session_data']['adaptive_assessment'] = assessment

            return {
                "assessment_id": assessment['id'],
                "type": "adaptive",
                "session_id": session_id,
                "total_questions": len(assessment['questions']),
                "difficulty": difficulty.value,
                "section_id": section_id
            }

        else:
            raise ValueError(f"Unknown assessment type: {assessment_type}")

    def submit_assessment_answer(self, user_id: str, assessment_id: str, question_id: str,
                               answer: Any, time_spent: int = None) -> Dict[str, Any]:
        """Submit an answer for an assessment question"""
        session_id = self._get_active_session(user_id)
        if not session_id:
            raise ValueError("No active user session found")

        # Determine assessment type and route accordingly
        if assessment_id in self.quiz_system.active_sessions:
            # Quiz system assessment
            result = self.quiz_system.submit_answer(assessment_id, question_id, answer, time_spent)

            # If quiz is completed, process results
            if result.get('is_completed'):
                final_results = self.quiz_system.end_session(assessment_id)
                self._process_completed_assessment(user_id, final_results)

            return result

        else:
            # Adaptive assessment
            session_data = self.active_user_sessions[session_id]['session_data']
            assessment = session_data.get('adaptive_assessment')

            if not assessment or assessment['id'] != assessment_id:
                raise ValueError("Assessment not found")

            # Process answer using assessment engine
            question = None
            for q in assessment['questions']:
                if q.id == question_id:
                    question = q
                    break

            if not question:
                raise ValueError("Question not found")

            is_correct, score, feedback = self.assessment_engine.evaluate_answer(question, answer)

            # Record answer
            if 'answers' not in assessment:
                assessment['answers'] = {}

            assessment['answers'][question_id] = {
                'answer': answer,
                'is_correct': is_correct,
                'score': score,
                'time_spent': time_spent or 0,
                'timestamp': datetime.now().isoformat()
            }

            # Check if assessment is completed
            completed_questions = len(assessment['answers'])
            total_questions = len(assessment['questions'])
            is_completed = completed_questions >= total_questions

            if is_completed:
                final_results = self._finalize_adaptive_assessment(user_id, assessment)
                self._process_completed_assessment(user_id, final_results)

            return {
                "is_correct": is_correct,
                "score": score,
                "max_score": question.points,
                "feedback": feedback,
                "explanation": question.explanation,
                "is_completed": is_completed,
                "current_score": sum(a['score'] for a in assessment['answers'].values()),
                "max_possible_score": sum(q.points for q in assessment['questions'])
            }

    def _finalize_adaptive_assessment(self, user_id: str, assessment: Dict) -> Dict[str, Any]:
        """Finalize an adaptive assessment and calculate results"""
        answers = assessment['answers']
        questions = assessment['questions']

        total_score = sum(a['score'] for a in answers.values())
        max_score = sum(q.points for q in questions)
        percentage = (total_score / max_score) * 100 if max_score > 0 else 0

        # Calculate skill gains
        skill_gains = self.assessment_engine._calculate_skill_gains(assessment['section_id'], [])
        for answer in answers.values():
            question = next((q for q in questions if q.id == answer.get('question_id')), None)
            if question:
                for skill in question.tags:
                    if skill in skill_gains:
                        skill_gains[skill] += (answer['score'] / question.points) * 20

        return {
            'assessment_id': assessment['id'],
            'user_id': user_id,
            'section_id': assessment.get('section_id', ''),
            'questions_answered': len(answers),
            'total_score': total_score,
            'max_score': max_score,
            'percentage': percentage,
            'completed_at': datetime.now(),
            'skill_gains': skill_gains,
            'answers': answers
        }

    def _process_completed_assessment(self, user_id: str, result: Dict[str, Any]):
        """Process a completed assessment and update all systems"""
        # Update progress tracking
        self.progress_tracker.record_assessment_result(user_id, result)

        # Update user data for achievement checking
        user_data = self._prepare_user_data_for_achievements(user_id)

        # Check for new achievements and certificates
        new_achievements = self.achievement_system.check_achievements(user_id, user_data)
        new_certificates = self.achievement_system.check_certificates(user_id, user_data)

        # Send notifications if enabled
        if self.config['achievement_notifications'] and (new_achievements or new_certificates):
            self._send_achievement_notifications(user_id, new_achievements, new_certificates)

        logger.info(f"Processed completed assessment for user {user_id}: {result['percentage']:.1f}%")

    def _send_achievement_notifications(self, user_id: str, achievements: List, certificates: List):
        """Send notifications for new achievements and certificates"""
        # In a real system, this would send emails, push notifications, etc.
        notification_message = f"Congratulations {user_id}! You've earned:\n"

        if achievements:
            notification_message += f"\n🏆 Achievements ({len(achievements)}):\n"
            for achievement in achievements:
                notification_message += f"  • {achievement.name} ({achievement.tier.value})\n"

        if certificates:
            notification_message += f"\n📜 Certificates ({len(certificates)}):\n"
            for certificate in certificates:
                notification_message += f"  • {certificate.name}\n"

        logger.info(f"Notification for {user_id}: {notification_message}")

    def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user dashboard data"""
        # Validate user session
        session_id = self._get_active_session(user_id)
        if not session_id:
            raise ValueError("No active user session found")

        # Get progress analytics
        try:
            analytics = self.progress_tracker.get_comprehensive_analytics(user_id)
            progress_data = {
                "overall_progress": analytics.overall_progress,
                "learning_velocity": analytics.learning_velocity,
                "knowledge_retention": analytics.knowledge_retention,
                "strengths": analytics.strengths,
                "weaknesses": analytics.weaknesses,
                "recommendations": analytics.recommendations
            }
        except Exception as e:
            logger.warning(f"Could not get analytics for user {user_id}: {e}")
            progress_data = {
                "overall_progress": 0,
                "learning_velocity": 0,
                "knowledge_retention": 0,
                "strengths": [],
                "weaknesses": [],
                "recommendations": []
            }

        # Get achievements and certificates
        user_achievements = self.achievement_system.get_user_achievements(user_id)
        user_certificates = self.achievement_system.get_user_certificates(user_id)

        # Get learning path recommendation
        try:
            learning_path = self.progress_tracker.generate_learning_path_recommendation(user_id)
            path_data = {
                "current_section": learning_path.current_section,
                "priority_topics": learning_path.priority_topics[:5],
                "estimated_time": str(learning_path.estimated_completion_time),
                "difficulty_adjustment": learning_path.difficulty_adjustment
            }
        except Exception as e:
            logger.warning(f"Could not get learning path for user {user_id}: {e}")
            path_data = {
                "current_section": "01_Foundational_Machine_Learning",
                "priority_topics": [],
                "estimated_time": "Unknown",
                "difficulty_adjustment": "maintain"
            }

        # Get recent activity
        recent_assessments = self.progress_tracker.user_data.get(user_id, {}).get('assessment_history', [])[-5:]
        recent_activity = []
        for assessment in recent_assessments:
            recent_activity.append({
                "type": "assessment",
                "section": assessment.get('section_id', 'Unknown'),
                "score": assessment['score'],
                "timestamp": assessment['timestamp']
            })

        # Generate progress visualizations
        visualizations = self.progress_tracker.generate_progress_visualization(user_id)

        return {
            "user_id": user_id,
            "session_id": session_id,
            "progress": progress_data,
            "achievements": {
                "total_points": user_achievements["total_points"],
                "achievement_count": user_achievements["achievement_count"],
                "recent_achievements": user_achievements["achievements"][:5]  # Last 5
            },
            "certificates": {
                "active_count": user_certificates["active_count"],
                "total_count": user_certificates["total_count"],
                "recent_certificates": user_certificates["certificates"][:3]  # Last 3
            },
            "learning_path": path_data,
            "recent_activity": recent_activity,
            "visualizations": visualizations,
            "quick_actions": self._generate_quick_actions(user_id)
        }

    def _generate_quick_actions(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate quick action recommendations for the user"""
        actions = []

        # Get user data
        user_data = self.progress_tracker.user_data.get(user_id, {})
        current_level = user_data.get('current_level', 1)
        recent_assessments = user_data.get('assessment_history', [])

        # Recommend practice based on performance
        if recent_assessments:
            avg_score = sum(a['score'] for a in recent_assessments[-5:]) / min(5, len(recent_assessments))
            if avg_score < 70:
                actions.append({
                    "type": "review",
                    "title": "Review Weak Areas",
                    "description": "Focus on topics where you scored below 70%",
                    "priority": "high"
                })

        # Recommend next section
        completed_sections = set(a.get('section_id', '') for a in recent_assessments if a['score'] >= 70)
        next_sections = self.progress_tracker._get_next_sections(completed_sections)
        if next_sections:
            actions.append({
                "type": "learn",
                "title": f"Start {next_sections[0].split('_')[-1].replace('_', ' ')}",
                "description": "Begin the next section in your learning journey",
                "priority": "medium"
            })

        # Recommend practice quiz
        if len(recent_assessments) < 3:
            actions.append({
                "type": "practice",
                "title": "Practice Quiz",
                "description": "Take a practice quiz to reinforce your learning",
                "priority": "medium"
            })

        # Recommend achievement goals
        user_achievements = self.achievement_system.get_user_achievements(user_id)
        if user_achievements["achievement_count"] < 5:
            actions.append({
                "type": "achievement",
                "title": "Earn More Badges",
                "description": "Complete assessments to unlock achievements",
                "priority": "low"
            })

        return actions

    def get_admin_dashboard(self) -> Dict[str, Any]:
        """Get administrative dashboard data"""
        # System statistics
        total_users = len(self.progress_tracker.user_data)
        total_assessments = sum(len(user.get('assessment_history', [])) for user in self.progress_tracker.user_data.values())

        # Achievement statistics
        achievement_stats = self.achievement_system.get_achievement_statistics()

        # Recent activity
        recent_activity = []
        for user_id, user_data in self.progress_tracker.user_data.items():
            recent_assessments = user_data.get('assessment_history', [])[-3:]
            for assessment in recent_assessments:
                recent_activity.append({
                    "user_id": user_id,
                    "type": "assessment",
                    "score": assessment['score'],
                    "timestamp": assessment['timestamp'],
                    "section": assessment.get('section_id', 'Unknown')
                })

        # Sort by timestamp and get recent
        recent_activity.sort(key=lambda x: x['timestamp'], reverse=True)
        recent_activity = recent_activity[:20]  # Last 20 activities

        # Generate leaderboard
        leaderboard = self.achievement_system.generate_leaderboard("overall", limit=10)

        # System health
        system_health = {
            "active_sessions": len(self.active_user_sessions),
            "total_questions": len(self.assessment_engine.questions),
            "total_achievements": len(self.achievement_system.achievements),
            "total_certificates": len(self.achievement_system.certificates),
            "data_retention_days": self.config['data_retention_days']
        }

        return {
            "system_overview": {
                "total_users": total_users,
                "total_assessments": total_assessments,
                "total_achievements_earned": achievement_stats['total_achievements_earned'],
                "total_certificates_earned": achievement_stats['total_certificates_earned']
            },
            "recent_activity": recent_activity,
            "leaderboard": leaderboard[:5],  # Top 5
            "system_health": system_health,
            "popular_sections": self._get_popular_sections(),
            "performance_metrics": self._get_performance_metrics()
        }

    def _get_popular_sections(self) -> List[Dict[str, Any]]:
        """Get most popular sections by assessment count"""
        section_counts = {}

        for user_data in self.progress_tracker.user_data.values():
            assessment_history = user_data.get('assessment_history', [])
            for assessment in assessment_history:
                section_id = assessment.get('section_id', 'Unknown')
                if section_id not in section_counts:
                    section_counts[section_id] = 0
                section_counts[section_id] += 1

        # Sort by count and get top sections
        popular_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return [
            {"section_id": section_id, "count": count}
            for section_id, count in popular_sections
        ]

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        all_scores = []
        all_times = []

        for user_data in self.progress_tracker.user_data.values():
            assessment_history = user_data.get('assessment_history', [])
            for assessment in assessment_history:
                all_scores.append(assessment['score'])
                all_times.append(assessment.get('time_taken', 0))

        metrics = {
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "average_time": sum(all_times) / len(all_times) if all_times else 0,
            "total_assessments": len(all_scores),
            "score_distribution": self._calculate_distribution(all_scores),
            "time_distribution": self._calculate_distribution(all_times)
        }

        return metrics

    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        """Calculate distribution of values"""
        if not values:
            return {}

        min_val, max_val = min(values), max(values)
        bins = 5
        bin_width = (max_val - min_val) / bins
        distribution = {}

        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            bin_label = f"{bin_start:.1f}-{bin_end:.1f}"
            distribution[bin_label] = sum(1 for v in values if bin_start <= v < bin_end)

        return distribution

    def _get_active_session(self, user_id: str) -> Optional[str]:
        """Get active session ID for user"""
        for session_id, session_data in self.active_user_sessions.items():
            if session_data['user_id'] == user_id:
                # Check if session is still active (within timeout)
                timeout_minutes = self.config['session_timeout_minutes']
                last_activity = session_data['last_activity']
                if (datetime.now() - last_activity).total_seconds() < timeout_minutes * 60:
                    # Update last activity
                    session_data['last_activity'] = datetime.now()
                    return session_id
                else:
                    # Remove expired session
                    del self.active_user_sessions[session_id]

        return None

    def logout_user(self, user_id: str) -> bool:
        """Logout user and end session"""
        session_id = self._get_active_session(user_id)
        if session_id:
            del self.active_user_sessions[session_id]
            logger.info(f"User {user_id} logged out successfully")
            return True
        return False

    def generate_report(self, user_id: str, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate various types of user reports"""
        if report_type == "comprehensive":
            return self.achievement_system.generate_progress_report(user_id)
        elif report_type == "achievements":
            return self.achievement_system.get_user_achievements(user_id)
        elif report_type == "certificates":
            return self.achievement_system.get_user_certificates(user_id)
        elif report_type == "analytics":
            try:
                analytics = self.progress_tracker.get_comprehensive_analytics(user_id)
                return asdict(analytics)
            except Exception as e:
                return {"error": str(e)}
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def cleanup_expired_data(self):
        """Clean up expired data based on retention policy"""
        retention_days = self.config['data_retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        # Clean up old assessment history
        for user_id, user_data in self.progress_tracker.user_data.items():
            assessment_history = user_data.get('assessment_history', [])
            recent_history = [
                assessment for assessment in assessment_history
                if datetime.fromisoformat(assessment['timestamp']) > cutoff_date
            ]
            user_data['assessment_history'] = recent_history

        self.progress_tracker.save_data()
        logger.info(f"Cleaned up data older than {retention_days} days")

    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }

        # Check each component
        try:
            # Assessment Engine
            health_status["components"]["assessment_engine"] = {
                "status": "healthy",
                "questions_count": len(self.assessment_engine.questions)
            }

            # Progress Tracker
            health_status["components"]["progress_tracker"] = {
                "status": "healthy",
                "users_count": len(self.progress_tracker.user_data)
            }

            # Quiz System
            health_status["components"]["quiz_system"] = {
                "status": "healthy",
                "questions_count": len(self.quiz_system.question_bank),
                "active_sessions": len(self.quiz_system.active_sessions)
            }

            # Achievement System
            health_status["components"]["achievement_system"] = {
                "status": "healthy",
                "achievements_count": len(self.achievement_system.achievements),
                "certificates_count": len(self.achievement_system.certificates)
            }

            # Data storage
            data_path_size = sum(f.stat().st_size for f in self.data_path.rglob('*') if f.is_file())
            health_status["components"]["data_storage"] = {
                "status": "healthy",
                "size_mb": round(data_path_size / (1024 * 1024), 2)
            }

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

if __name__ == "__main__":
    # Example usage
    dashboard = AssessmentSystemDashboard()

    # User login
    login_result = dashboard.user_login("demo_user", {
        "display_name": "Demo User",
        "experience_level": "intermediate"
    })
    print(f"Login result: {login_result}")

    # Start an assessment
    assessment_config = {
        "type": "quiz",
        "mode": "practice",
        "section_id": "01_Foundational_Machine_Learning",
        "difficulty": 2,
        "num_questions": 5,
        "time_limit": 600  # 10 minutes
    }

    assessment_result = dashboard.start_assessment("demo_user", assessment_config)
    print(f"Assessment started: {assessment_result}")

    # Get user dashboard
    try:
        user_dashboard = dashboard.get_user_dashboard("demo_user")
        print(f"User dashboard retrieved successfully")
        print(f"Overall progress: {user_dashboard['progress']['overall_progress']:.1f}%")
        print(f"Achievements: {user_dashboard['achievements']['achievement_count']}")
        print(f"Certificates: {user_dashboard['certificates']['active_count']}")
    except Exception as e:
        print(f"Error getting dashboard: {e}")

    # Get admin dashboard
    try:
        admin_dashboard = dashboard.get_admin_dashboard()
        print(f"\nAdmin Dashboard:")
        print(f"Total users: {admin_dashboard['system_overview']['total_users']}")
        print(f"Total assessments: {admin_dashboard['system_overview']['total_assessments']}")
        print(f"Active sessions: {admin_dashboard['system_health']['active_sessions']}")
    except Exception as e:
        print(f"Error getting admin dashboard: {e}")

    # Health check
    health = dashboard.health_check()
    print(f"\nSystem Health: {health['status']}")
    for component, status in health['components'].items():
        print(f"  {component}: {status['status']}")

    # Logout
    logout_result = dashboard.logout_user("demo_user")
    print(f"Logout result: {logout_result}")

    print("\nAssessment System Dashboard initialized successfully!")