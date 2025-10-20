#!/usr/bin/env python3
"""
Interactive Learning System for AI Documentation
Creates comprehensive learning features including quizzes, assessments, and progress tracking
"""

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import random
import re
import statistics

@dataclass
class QuizQuestion:
    """Represents a quiz question"""
    id: str
    section_id: str
    question_type: str  # multiple_choice, true_false, fill_blank, code_execution
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: int  # 1-5
    topic: str
    prerequisites: List[str] = None

@dataclass
class LearningProgress:
    """Tracks user learning progress"""
    user_id: str
    section_id: str
    completion_percentage: float
    time_spent: int  # minutes
    last_accessed: datetime
    quiz_scores: List[float]
    badges_earned: List[str]
    notes_taken: int
    bookmarks_count: int
    concept_mastery: Dict[str, float]

@dataclass
class Achievement:
    """Represents an achievement/badge"""
    id: str
    name: str
    description: str
    icon: str
    requirements: Dict[str, Any]
    points: int
    category: str

class InteractiveLearningSystem:
    """Main interactive learning system"""

    def __init__(self, docs_path: str = "/Users/dtumkorkmaz/Projects/ai-docs"):
        self.docs_path = Path(docs_path)
        self.quizzes = {}
        self.achievements = {}
        self.learning_paths = {}
        self.user_progress = {}

        # Initialize system
        self._load_achievements()
        self._create_learning_paths()
        self._generate_sample_quizzes()

    def _load_achievements(self):
        """Load achievement definitions"""
        self.achievements = {
            "first_steps": Achievement(
                id="first_steps",
                name="First Steps",
                description="Complete your first section",
                icon="👣",
                requirements={"sections_completed": 1},
                points=10,
                category="progress"
            ),
            "quick_learner": Achievement(
                id="quick_learner",
                name="Quick Learner",
                description="Complete a section in under 30 minutes",
                icon="⚡",
                requirements={"time_limit": 30, "sections_completed": 1},
                points=25,
                category="speed"
            ),
            "perfectionist": Achievement(
                id="perfectionist",
                name="Perfectionist",
                description="Score 100% on 5 quizzes in a row",
                icon="💯",
                requirements={"perfect_quizzes": 5},
                points=50,
                category="excellence"
            ),
            "consistent_learner": Achievement(
                id="consistent_learner",
                name="Consistent Learner",
                description="Study for 7 days in a row",
                icon="📅",
                requirements={"consecutive_days": 7},
                points=30,
                category="consistency"
            ),
            "knowledge_collector": Achievement(
                id="knowledge_collector",
                name="Knowledge Collector",
                description="Complete 10 sections",
                icon="📚",
                requirements={"sections_completed": 10},
                points=40,
                category="volume"
            ),
            "ai_expert": Achievement(
                id="ai_expert",
                name="AI Expert",
                description="Complete all 25 sections",
                icon="🎓",
                requirements={"sections_completed": 25},
                points=100,
                category="mastery"
            ),
            "night_owl": Achievement(
                id="night_owl",
                name="Night Owl",
                description="Study after 10 PM for 5 days",
                icon="🦉",
                requirements={"late_sessions": 5},
                points=20,
                category="time"
            ),
            "early_bird": Achievement(
                id="early_bird",
                name="Early Bird",
                description="Study before 8 AM for 5 days",
                icon="🐦",
                requirements={"early_sessions": 5},
                points=20,
                category="time"
            ),
            "explorer": Achievement(
                id="explorer",
                name="Explorer",
                description="Try all 5 learning paths",
                icon="🧭",
                requirements={"paths_tried": 5},
                points=35,
                category="exploration"
            ),
            "note_taker": Achievement(
                id="note_taker",
                name="Note Taker",
                description="Take 50 notes across sections",
                icon="📝",
                requirements={"total_notes": 50},
                points=25,
                category="engagement"
            )
        }

    def _create_learning_paths(self):
        """Create predefined learning paths"""
        self.learning_paths = {
            "beginner": {
                "name": "AI Fundamentals",
                "description": "Perfect for beginners new to AI",
                "difficulty": "Beginner",
                "estimated_hours": 40,
                "sections": [
                    "01_Foundational_Machine_Learning",
                    "02_Advanced_Deep_Learning",
                    "03_Natural_Language_Processing"
                ],
                "prerequisites": [],
                "color": "#4CAF50"
            },
            "intermediate": {
                "name": "AI Developer Path",
                "description": "For developers wanting to build AI applications",
                "difficulty": "Intermediate",
                "estimated_hours": 80,
                "sections": [
                    "04_Computer_Vision",
                    "05_Generative_AI",
                    "06_AI_Agents_and_Autonomous",
                    "14_MLOps_and_AI_Deployment_Strategies"
                ],
                "prerequisites": ["beginner"],
                "color": "#2196F3"
            },
            "advanced": {
                "name": "AI Research Path",
                "description": "For researchers and advanced AI practitioners",
                "difficulty": "Advanced",
                "estimated_hours": 60,
                "sections": [
                    "12_Emerging_Research_2025",
                    "15_State_Space_Models_and_Mamba_Architecture",
                    "16_Advanced_Multimodal_AI_Integration",
                    "25_Future_of_AI_and_Emerging_Trends"
                ],
                "prerequisites": ["intermediate"],
                "color": "#9C27B0"
            },
            "researcher": {
                "name": "Academic AI Path",
                "description": "Academic focus with theoretical foundations",
                "difficulty": "Advanced",
                "estimated_hours": 120,
                "sections": [
                    "01_Foundational_Machine_Learning",
                    "07_AI_Ethics_and_Safety",
                    "12_Emerging_Research_2025",
                    "19_Human_AI_Collaboration_and_Augmentation"
                ],
                "prerequisites": [],
                "color": "#FF5722"
            },
            "practitioner": {
                "name": "AI Engineering Path",
                "description": "Practical engineering focus",
                "difficulty": "Intermediate",
                "estimated_hours": 150,
                "sections": [
                    "02_Advanced_Deep_Learning",
                    "06_AI_Agents_and_Autonomous",
                    "08_AI_Applications_Industry",
                    "14_MLOps_and_AI_Deployment_Strategies",
                    "22_AI_for_Smart_Cities_and_Infrastructure"
                ],
                "prerequisites": ["beginner"],
                "color": "#607D8B"
            },
            "industry": {
                "name": "Industry Professional Path",
                "description": "Business and industry applications",
                "difficulty": "Intermediate",
                "estimated_hours": 45,
                "sections": [
                    "08_AI_Applications_Industry",
                    "14_AI_Business_Enterprise",
                    "17_AI_for_Social_Good_and_Impact",
                    "18_AI_Policy_and_Regulation"
                ],
                "prerequisites": [],
                "color": "#FF9800"
            }
        }

    def _generate_sample_quizzes(self):
        """Generate sample quiz questions for key sections"""
        self.quizzes = {
            "01_Foundational_Machine_Learning": [
                QuizQuestion(
                    id="ml_001",
                    section_id="01_Foundational_Machine_Learning",
                    question_type="multiple_choice",
                    question="What is the primary goal of supervised learning?",
                    options=[
                        "To find hidden patterns in unlabeled data",
                        "To learn a mapping function from labeled examples",
                        "To maximize reward through trial and error",
                        "To reduce the dimensionality of data"
                    ],
                    correct_answer="To learn a mapping function from labeled examples",
                    explanation="Supervised learning learns from labeled examples to map inputs to outputs",
                    difficulty=2,
                    topic="supervised_learning"
                ),
                QuizQuestion(
                    id="ml_002",
                    section_id="01_Foundational_Machine_Learning",
                    question_type="true_false",
                    question="Linear regression can only model linear relationships.",
                    options=["True", "False"],
                    correct_answer="False",
                    explanation="Linear regression can model linear relationships in the parameters, but can capture non-linear relationships in the input features through transformations.",
                    difficulty=3,
                    topic="linear_regression"
                )
            ],
            "03_Natural_Language_Processing": [
                QuizQuestion(
                    id="nlp_001",
                    section_id="03_Natural_Language_Processing",
                    question_type="multiple_choice",
                    question="What is the main advantage of transformer models over RNNs?",
                    options=[
                        "They are faster to train",
                        "They can process all tokens in parallel",
                        "They use less memory",
                        "They only work with short sequences"
                    ],
                    correct_answer="They can process all tokens in parallel",
                    explanation="Transformers use self-attention mechanisms that allow parallel processing of all tokens, unlike sequential RNNs.",
                    difficulty=3,
                    topic="transformers"
                )
            ],
            "04_Computer_Vision": [
                QuizQuestion(
                    id="cv_001",
                    section_id="04_Computer_Vision",
                    question_type="multiple_choice",
                    question="What is the purpose of convolution in CNNs?",
                    options=[
                        "To reduce overfitting",
                        "To extract spatial features",
                        "To normalize the data",
                        "To classify images"
                    ],
                    correct_answer="To extract spatial features",
                    explanation="Convolution layers apply filters to extract spatial features like edges, textures, and patterns from images.",
                    difficulty=2,
                    topic="convolution"
                )
            ]
        }

    def get_quiz_for_section(self, section_id: str, difficulty: int = 3) -> List[QuizQuestion]:
        """Get quiz questions for a specific section"""
        if section_id in self.quizzes:
            return [q for q in self.quizzes[section_id] if q.difficulty <= difficulty]
        return []

    def generate_code_exercise(self, section_id: str, topic: str) -> Dict[str, Any]:
        """Generate a code exercise for a specific section and topic"""
        exercises = {
            "01_Foundational_Machine_Learning": {
                "linear_regression": {
                    "title": "Linear Regression from Scratch",
                    "description": "Implement linear regression using gradient descent",
                    "difficulty": 3,
                    "template": '''import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Your implementation here
        pass

    def predict(self, X):
        # Your implementation here
        pass

# Test your implementation
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print("Predictions:", predictions)
print("R² Score:", r2_score(y, predictions))''',
                    "solution": '''def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

def predict(self, X):
        return np.dot(X, self.weights) + self.bias'''
                }
            },
            "03_Natural_Language_Processing": {
                "text_preprocessing": {
                    "title": "Text Preprocessing Pipeline",
                    "description": "Create a text preprocessing pipeline for NLP",
                    "difficulty": 2,
                    "template": '''import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Your implementation here
        pass

# Test your implementation
text = "The quick brown fox jumps over the lazy dog. Natural Language Processing is amazing!"
processor = TextPreprocessor()
processed = processor.preprocess_text(text)
print("Original:", text)
print("Processed:", processed)''',
                    "solution": '''def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and stem
        tokens = [self.stemmer.stem(token) for token in tokens
                  if token not in self.stop_words]

        return tokens'''
                }
            }
        }

        if section_id in exercises and topic in exercises[section_id]:
            return exercises[section_id][topic]
        return None

    def check_user_progress(self, user_id: str, section_id: str) -> LearningProgress:
        """Check and update user progress"""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}

        if section_id not in self.user_progress[user_id]:
            self.user_progress[user_id][section_id] = LearningProgress(
                user_id=user_id,
                section_id=section_id,
                completion_percentage=0.0,
                time_spent=0,
                last_accessed=datetime.now(),
                quiz_scores=[],
                badges_earned=[],
                notes_taken=0,
                bookmarks_count=0,
                concept_mastery={}
            )

        return self.user_progress[user_id][section_id]

    def update_progress(self, user_id: str, section_id: str, updates: Dict[str, Any]):
        """Update user progress"""
        progress = self.check_user_progress(user_id, section_id)

        for key, value in updates.items():
            if hasattr(progress, key):
                setattr(progress, key, value)

        progress.last_accessed = datetime.now()

        # Check for new achievements
        new_badges = self.check_achievements(user_id)
        if new_badges:
            progress.badges_earned.extend(new_badges)

        return progress

    def check_achievements(self, user_id: str) -> List[str]:
        """Check and award new achievements"""
        new_badges = []
        user_data = self.user_progress.get(user_id, {})

        for achievement_id, achievement in self.achievements.items():
            # Skip if already earned
            if any(achievement_id in prog.badges_earned for prog in user_data.values()):
                continue

            # Check requirements
            if self._check_achievement_requirements(user_id, achievement):
                new_badges.append(achievement_id)

        return new_badges

    def _check_achievement_requirements(self, user_id: str, achievement: Achievement) -> bool:
        """Check if user meets achievement requirements"""
        user_data = self.user_progress.get(user_id, {})

        if achievement.requirements.get("sections_completed", 0) > 0:
            completed_sections = sum(1 for prog in user_data.values()
                                  if prog.completion_percentage >= 100)
            if completed_sections < achievement.requirements["sections_completed"]:
                return False

        if achievement.requirements.get("perfect_quizzes", 0) > 0:
            all_scores = []
            for prog in user_data.values():
                all_scores.extend(prog.quiz_scores)

            perfect_count = sum(1 for score in all_scores if score == 100)
            if perfect_count < achievement.requirements["perfect_quizzes"]:
                return False

        if achievement.requirements.get("consecutive_days", 0) > 0:
            # This would require date tracking - simplified for demo
            return False

        return True

    def generate_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate personalized learning recommendations"""
        user_data = self.user_progress.get(user_id, {})

        recommendations = []

        # Find sections with progress > 0 but < 100
        for section_id, progress in user_data.items():
            if 0 < progress.completion_percentage < 100:
                recommendations.append({
                    "type": "continue_learning",
                    "section_id": section_id,
                    "message": f"Continue learning {section_id}",
                    "priority": "high"
                })

        # Find completed sections with quiz scores < 80
        for section_id, progress in user_data.items():
            if progress.completion_percentage >= 100:
                avg_score = np.mean(progress.quiz_scores) if progress.quiz_scores else 0
                if avg_score < 80:
                    recommendations.append({
                        "type": "review_needed",
                        "section_id": section_id,
                        "message": f"Review {section_id} - average score: {avg_score:.1f}%",
                        "priority": "medium"
                    })

        # Suggest new sections based on completed prerequisites
        completed_sections = [section_id for section_id, progress in user_data.items()
                             if progress.completion_percentage >= 100]

        for path_name, path_data in self.learning_paths.items():
            if path_data.get("prerequisites"):
                prereqs_completed = all(prereq in completed_sections for prereq in path_data["prerequisites"])
                if prereqs_completed and path_name not in [p.get("current_path") for p in recommendations]:
                    recommendations.append({
                        "type": "new_learning_path",
                        "path_name": path_name,
                        "message": f"Ready for {path_data['name']} path",
                        "priority": "high"
                    })

        return recommendations[:5]  # Return top 5 recommendations

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data for backup/analysis"""
        user_data = self.user_progress.get(user_id, {})

        export_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "progress": user_data,
            "achievements": [],
            "statistics": {
                "total_sections": len(user_data),
                "completed_sections": sum(1 for prog in user_data.values() if prog.completion_percentage >= 100),
                "total_time_spent": sum(prog.time_spent for prog in user_data.values()),
                "total_quizzes_taken": sum(len(prog.quiz_scores) for prog in user_data.values()),
                "average_quiz_score": np.mean([score for prog in user_data.values() for score in prog.quiz_scores]) if any(prog.quiz_scores for prog in user_data.values()) else 0,
                "total_notes_taken": sum(prog.notes_taken for prog in user_data.values()),
                "total_bookmarks": sum(prog.bookmarks_count for prog in user_data.values())
            }
        }

        # Collect achievements
        for prog in user_data.values():
            export_data["achievements"].extend(prog.badges_earned)

        export_data["achievements"] = list(set(export_data["achievements"]))

        return export_data

    def save_user_data(self, user_id: str, data: Dict[str, Any]):
        """Save imported user data"""
        if "progress" in data:
            self.user_progress[user_id] = data["progress"]

    def get_learning_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive learning analytics"""
        user_data = self.user_progress.get(user_id, {})

        if not user_data:
            return {"error": "No data available for this user"}

        # Calculate statistics
        total_sections = len(user_data)
        completed_sections = sum(1 for prog in user_data.values() if prog.completion_percentage >= 100)
        overall_progress = (completed_sections / total_sections * 100) if total_sections > 0 else 0

        # Time analytics
        total_time = sum(prog.time_spent for prog in user_data.values())
        avg_time_per_section = total_time / total_sections if total_sections > 0 else 0

        # Quiz analytics
        all_scores = [score for prog in user_data.values() for score in prog.quiz_scores]
        avg_quiz_score = statistics.mean(all_scores) if all_scores else 0
        quiz_count = len(all_scores)

        # Achievement analytics
        all_badges = list(set([badge for prog in user_data.values() for badge in prog.badges_earned]))
        achievement_count = len(all_badges)

        # Learning streak (simplified)
        access_dates = [prog.last_accessed.date() for prog in user_data.values()]
        unique_dates = len(set(access_dates))

        return {
            "user_id": user_id,
            "overall_progress": overall_progress,
            "sections_completed": completed_sections,
            "total_sections": total_sections,
            "total_time_spent_minutes": total_time,
            "average_time_per_section_minutes": avg_time_per_section,
            "quiz_statistics": {
                "total_quizzes_taken": quiz_count,
                "average_score": avg_quiz_score,
                "highest_score": max(all_scores) if all_scores else 0,
                "perfect_scores": sum(1 for score in all_scores if score == 100)
            },
            "achievement_statistics": {
                "total_achievements": achievement_count,
                "achievement_badges": all_badges,
                "total_points": sum(self.achievements[badge].points for badge in all_badges if badge in self.achievements)
            },
            "engagement_statistics": {
                "total_notes_taken": sum(prog.notes_taken for prog in user_data.values()),
                "total_bookmarks": sum(prog.bookmarks_count for prog in user_data.values()),
                "unique_study_days": unique_dates
            },
            "learning_recommendations": self.generate_recommendations(user_id)
        }

# Main function to create interactive learning components
def create_interactive_learning_system():
    """Create and return the interactive learning system"""
    return InteractiveLearningSystem()

# Example usage
if __name__ == "__main__":
    # Create the learning system
    learning_system = create_interactive_learning_system()

    print("🎓 Interactive Learning System Initialized")
    print(f"📚 Available Learning Paths: {len(learning_system.learning_paths)}")
    print(f"🏆 Available Achievements: {len(learning_system.achievements)}")
    print(f"📝 Quiz Questions: {sum(len(questions) for questions in learning_system.quizzes.values())}")

    # Generate sample user data
    user_id = "demo_user"
    progress = learning_system.check_user_progress(user_id, "01_Foundational_Machine_Learning")
    learning_system.update_progress(user_id, "01_Foundational_Machine_Learning", {
        "completion_percentage": 75,
        "time_spent": 45,
        "quiz_scores": [85, 92, 78],
        "notes_taken": 5,
        "bookmarks_count": 3
    })

    # Get analytics
    analytics = learning_system.get_learning_analytics(user_id)
    print(f"\n📊 Analytics for {user_id}:")
    print(f"   Overall Progress: {analytics['overall_progress']:.1f}%")
    print(f"   Sections Completed: {analytics['sections_completed']}/{analytics['total_sections']}")
    print(f"   Time Spent: {analytics['total_time_spent_minutes']} minutes")
    print(f"   Average Quiz Score: {analytics['quiz_statistics']['average_score']:.1f}%")

    # Get recommendations
    recommendations = learning_system.generate_recommendations(user_id)
    print(f"\n💡 Recommendations:")
    for rec in recommendations:
        print(f"   • {rec['message']}")