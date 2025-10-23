#!/usr/bin/env python3
"""
Core Assessment Engine for AI Documentation Project
Handles assessment creation, evaluation, and progress tracking
"""

import json
import uuid
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path

class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    CODING_CHALLENGE = "coding_challenge"
    ESSAY = "essay"
    CASE_STUDY = "case_study"
    PROOF = "proof"
    DESIGN_PROBLEM = "design_problem"

class DifficultyLevel(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5

class AssessmentType(Enum):
    FORMATIVE = "formative"  # Learning assessments
    SUMMATIVE = "summative"  # Evaluation assessments
    DIAGNOSTIC = "diagnostic"  # Skill gap identification
    CERTIFICATION = "certification"  # Professional credentials

@dataclass
class Question:
    """Base question structure"""
    id: str
    question_type: QuestionType
    title: str
    description: str
    difficulty: DifficultyLevel
    section_id: str
    subsection: str
    content: Dict[str, Any]
    correct_answer: Any
    points: int
    time_limit: Optional[int] = None  # in seconds
    hints: List[str] = None
    explanation: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.hints is None:
            self.hints = []
        if self.tags is None:
            self.tags = []

@dataclass
class UserAnswer:
    """User response to a question"""
    question_id: str
    answer: Any
    timestamp: datetime
    time_spent: int  # in seconds
    hints_used: int = 0
    attempts: int = 1
    is_correct: bool = False
    score: float = 0.0
    feedback: str = ""

@dataclass
class AssessmentResult:
    """Complete assessment results"""
    assessment_id: str
    user_id: str
    questions_answered: List[UserAnswer]
    total_score: float
    max_score: float
    percentage: float
    time_taken: int
    completed_at: datetime
    skill_gains: Dict[str, float]
    recommendations: List[str]

class AssessmentEngine:
    """Main assessment engine for creating and evaluating assessments"""

    def __init__(self, data_path: str = "assessment_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        self.questions = {}
        self.assessments = {}
        self.user_profiles = {}
        self.progress_records = {}

        self.load_data()

    def load_data(self):
        """Load existing assessment data"""
        # Load questions
        questions_file = self.data_path / "questions.json"
        if questions_file.exists():
            with open(questions_file, 'r') as f:
                questions_data = json.load(f)
                for qid, qdata in questions_data.items():
                    self.questions[qid] = self._dict_to_question(qdata)

        # Load user profiles
        profiles_file = self.data_path / "user_profiles.json"
        if profiles_file.exists():
            with open(profiles_file, 'r') as f:
                self.user_profiles = json.load(f)

    def save_data(self):
        """Save assessment data"""
        # Save questions
        questions_file = self.data_path / "questions.json"
        questions_data = {qid: self._question_to_dict(q) for qid, q in self.questions.items()}
        with open(questions_file, 'w') as f:
            json.dump(questions_data, f, indent=2, default=str)

        # Save user profiles
        profiles_file = self.data_path / "user_profiles.json"
        with open(profiles_file, 'w') as f:
            json.dump(self.user_profiles, f, indent=2, default=str)

    def create_question(self, **kwargs) -> Question:
        """Create a new question"""
        question_id = kwargs.get('id', str(uuid.uuid4()))

        question = Question(
            id=question_id,
            question_type=QuestionType(kwargs['question_type']),
            title=kwargs['title'],
            description=kwargs['description'],
            difficulty=DifficultyLevel(kwargs['difficulty']),
            section_id=kwargs['section_id'],
            subsection=kwargs.get('subsection', ''),
            content=kwargs['content'],
            correct_answer=kwargs['correct_answer'],
            points=kwargs.get('points', 10),
            time_limit=kwargs.get('time_limit'),
            hints=kwargs.get('hints', []),
            explanation=kwargs.get('explanation', ''),
            tags=kwargs.get('tags', [])
        )

        self.questions[question_id] = question
        self.save_data()
        return question

    def evaluate_answer(self, question: Question, user_answer: Any) -> Tuple[bool, float, str]:
        """Evaluate a user's answer and return correctness, score, and feedback"""

        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            is_correct = str(user_answer).strip().lower() == str(question.correct_answer).strip().lower()
            score = question.points if is_correct else 0
            feedback = question.explanation if is_correct else f"Correct answer: {question.correct_answer}"

        elif question.question_type == QuestionType.TRUE_FALSE:
            is_correct = str(user_answer).strip().lower() == str(question.correct_answer).strip().lower()
            score = question.points if is_correct else 0
            feedback = question.explanation if is_correct else f"Correct answer: {question.correct_answer}"

        elif question.question_type == QuestionType.FILL_BLANK:
            # More flexible matching for fill-in-blank
            correct_answers = question.correct_answer if isinstance(question.correct_answer, list) else [question.correct_answer]
            is_correct = any(str(user_answer).strip().lower() == str(ans).strip().lower() for ans in correct_answers)
            score = question.points if is_correct else 0
            feedback = question.explanation if is_correct else f"Possible answers: {', '.join(correct_answers)}"

        elif question.question_type == QuestionType.CODING_CHALLENGE:
            # Code evaluation (would integrate with actual code execution)
            is_correct, score, feedback = self._evaluate_code_challenge(question, user_answer)

        elif question.question_type == QuestionType.ESSAY:
            # Essay evaluation (would integrate with NLP analysis)
            is_correct, score, feedback = self._evaluate_essay(question, user_answer)

        else:
            # Default evaluation
            is_correct = str(user_answer) == str(question.correct_answer)
            score = question.points if is_correct else 0
            feedback = question.explanation if is_correct else "Answer not correct"

        return is_correct, score, feedback

    def _evaluate_code_challenge(self, question: Question, user_code: str) -> Tuple[bool, float, str]:
        """Evaluate coding challenge answers"""
        # Placeholder for actual code execution and testing
        # In real implementation, this would:
        # 1. Execute the code in a sandbox
        # 2. Run test cases
        # 3. Check output correctness
        # 4. Evaluate code quality

        test_cases = question.content.get('test_cases', [])
        if not test_cases:
            return False, 0, "No test cases defined"

        # Simulated evaluation
        passed_tests = random.randint(0, len(test_cases))
        score = (passed_tests / len(test_cases)) * question.points

        feedback = f"Passed {passed_tests}/{len(test_cases)} test cases"
        if passed_tests == len(test_cases):
            feedback += ". Excellent work!"
        elif passed_tests > 0:
            feedback += f". {len(test_cases) - passed_tests} tests failed."
        else:
            feedback += ". All tests failed. Review the implementation."

        return passed_tests == len(test_cases), score, feedback

    def _evaluate_essay(self, question: Question, essay: str) -> Tuple[bool, float, str]:
        """Evaluate essay answers"""
        # Placeholder for NLP-based essay evaluation
        # In real implementation, this would:
        # 1. Analyze essay structure and coherence
        # 2. Check for key concepts and understanding
        # 3. Evaluate depth of analysis
        # 4. Assess writing quality

        essay_length = len(essay.split())
        min_length = question.content.get('min_length', 100)

        if essay_length < min_length:
            return False, 0, f"Essay too short. Minimum {min_length} words required."

        # Simulated scoring based on key concept coverage
        key_concepts = question.content.get('key_concepts', [])
        covered_concepts = sum(1 for concept in key_concepts if concept.lower() in essay.lower())

        score = min(question.points, (covered_concepts / len(key_concepts)) * question.points)
        is_correct = score >= (question.points * 0.7)  # 70% threshold for passing

        feedback = f"Essay covers {covered_concepts}/{len(key_concepts)} key concepts"
        if is_correct:
            feedback += ". Good understanding demonstrated."
        else:
            feedback += ". Consider including more key concepts."

        return is_correct, score, feedback

    def generate_assessment(self, section_id: str, difficulty: DifficultyLevel,
                          num_questions: int, assessment_type: AssessmentType) -> Dict:
        """Generate a customized assessment"""

        # Filter questions by section and difficulty
        section_questions = [q for q in self.questions.values()
                           if q.section_id == section_id and q.difficulty == difficulty]

        if len(section_questions) < num_questions:
            # Adjust difficulty if not enough questions
            available_difficulties = [q.difficulty for q in self.questions.values() if q.section_id == section_id]
            if available_difficulties:
                # Use closest available difficulty
                closest_difficulty = min(available_difficulties, key=lambda x: abs(x.value - difficulty.value))
                section_questions = [q for q in self.questions.values()
                                   if q.section_id == section_id and q.difficulty == closest_difficulty]

        # Randomly select questions
        selected_questions = random.sample(section_questions, min(num_questions, len(section_questions)))

        assessment = {
            'id': str(uuid.uuid4()),
            'type': assessment_type.value,
            'section_id': section_id,
            'difficulty': difficulty.value,
            'questions': selected_questions,
            'total_points': sum(q.points for q in selected_questions),
            'created_at': datetime.now()
        }

        self.assessments[assessment['id']] = assessment
        return assessment

    def submit_assessment(self, user_id: str, assessment_id: str, answers: Dict[str, Any]) -> AssessmentResult:
        """Process assessment submission"""

        if assessment_id not in self.assessments:
            raise ValueError(f"Assessment {assessment_id} not found")

        assessment = self.assessments[assessment_id]
        user_answers = []
        total_score = 0
        max_score = assessment['total_points']

        start_time = datetime.now()

        for question in assessment['questions']:
            if question.id in answers:
                user_answer_data = answers[question.id]

                is_correct, score, feedback = self.evaluate_answer(question, user_answer_data['answer'])

                user_answer = UserAnswer(
                    question_id=question.id,
                    answer=user_answer_data['answer'],
                    timestamp=datetime.now(),
                    time_spent=user_answer_data.get('time_spent', 0),
                    hints_used=user_answer_data.get('hints_used', 0),
                    attempts=user_answer_data.get('attempts', 1),
                    is_correct=is_correct,
                    score=score,
                    feedback=feedback
                )

                user_answers.append(user_answer)
                total_score += score

        # Calculate skill gains and recommendations
        skill_gains = self._calculate_skill_gains(assessment['section_id'], user_answers)
        recommendations = self._generate_recommendations(assessment['section_id'], user_answers)

        result = AssessmentResult(
            assessment_id=assessment_id,
            user_id=user_id,
            questions_answered=user_answers,
            total_score=total_score,
            max_score=max_score,
            percentage=(total_score / max_score) * 100,
            time_taken=int((datetime.now() - start_time).total_seconds()),
            completed_at=datetime.now(),
            skill_gains=skill_gains,
            recommendations=recommendations
        )

        # Update user profile
        self._update_user_profile(user_id, result)

        # Save progress
        if user_id not in self.progress_records:
            self.progress_records[user_id] = []
        self.progress_records[user_id].append(asdict(result))

        self.save_data()
        return result

    def _calculate_skill_gains(self, section_id: str, answers: List[UserAnswer]) -> Dict[str, float]:
        """Calculate skill improvements based on assessment performance"""

        # Define skill areas for each section
        skill_areas = {
            '01_Foundational_Machine_Learning': [
                'linear_algebra', 'calculus', 'probability', 'statistics', 'ml_algorithms'
            ],
            '02_Advanced_Deep_Learning': [
                'neural_networks', 'optimization', 'regularization', 'architecture_design'
            ],
            '03_Natural_Language_Processing': [
                'text_processing', 'language_models', 'nlp_algorithms', 'transformers'
            ],
            '04_Computer_Vision': [
                'image_processing', 'feature_extraction', 'deep_learning_cv', '3d_vision'
            ],
            '05_Generative_AI': [
                'generative_models', 'creativity', 'foundation_models', 'prompting'
            ]
        }

        skill_gains = {}
        if section_id in skill_areas:
            for skill in skill_areas[section_id]:
                # Calculate skill gain based on related question performance
                skill_score = sum(ans.score for ans in answers if skill in self.questions[ans.question_id].tags)
                skill_max = sum(self.questions[ans.question_id].points for ans in answers
                              if skill in self.questions[ans.question_id].tags)

                if skill_max > 0:
                    skill_gains[skill] = (skill_score / skill_max) * 100
                else:
                    skill_gains[skill] = 0

        return skill_gains

    def _generate_recommendations(self, section_id: str, answers: List[UserAnswer]) -> List[str]:
        """Generate personalized learning recommendations"""

        recommendations = []
        incorrect_answers = [ans for ans in answers if not ans.is_correct]

        if incorrect_answers:
            # Analyze common mistakes
            for answer in incorrect_answers:
                question = self.questions[answer.question_id]

                if question.question_type == QuestionType.CODING_CHALLENGE:
                    recommendations.append(f"Practice coding challenges in {question.subsection}")
                elif question.question_type == QuestionType.MULTIPLE_CHOICE:
                    recommendations.append(f"Review theoretical concepts in {question.subsection}")
                elif question.question_type == QuestionType.ESSAY:
                    recommendations.append(f"Improve analytical writing for {question.subsection}")

        # General recommendations based on performance
        percentage = sum(ans.score for ans in answers) / sum(self.questions[ans.question_id].points for ans in answers) * 100

        if percentage < 60:
            recommendations.append("Review foundational concepts before advancing")
        elif percentage < 80:
            recommendations.append("Focus on practical implementation exercises")
        else:
            recommendations.append("Ready for more advanced challenges")

        return recommendations

    def _update_user_profile(self, user_id: str, result: AssessmentResult):
        """Update user profile with assessment results"""

        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'assessments_completed': 0,
                'total_score': 0,
                'skills': {},
                'achievements': [],
                'level': 1,
                'experience_points': 0
            }

        profile = self.user_profiles[user_id]

        # Update basic stats
        profile['assessments_completed'] += 1
        profile['total_score'] += result.total_score
        profile['last_assessment'] = result.completed_at.isoformat()

        # Update skills
        for skill, gain in result.skill_gains.items():
            if skill not in profile['skills']:
                profile['skills'][skill] = {'level': 0, 'experience': 0}
            profile['skills'][skill]['experience'] += gain

            # Update skill level based on experience
            if profile['skills'][skill]['experience'] >= 100:
                profile['skills'][skill]['level'] = 5
            elif profile['skills'][skill]['experience'] >= 80:
                profile['skills'][skill]['level'] = 4
            elif profile['skills'][skill]['experience'] >= 60:
                profile['skills'][skill]['level'] = 3
            elif profile['skills'][skill]['experience'] >= 40:
                profile['skills'][skill]['level'] = 2
            elif profile['skills'][skill]['experience'] >= 20:
                profile['skills'][skill]['level'] = 1

        # Update overall level and experience
        profile['experience_points'] += int(result.percentage)
        profile['level'] = min(30, 1 + profile['experience_points'] // 1000)

        # Check for achievements
        self._check_achievements(user_id, result)

    def _check_achievements(self, user_id: str, result: AssessmentResult):
        """Check and award achievements based on performance"""

        profile = self.user_profiles[user_id]
        new_achievements = []

        # First assessment
        if profile['assessments_completed'] == 1:
            new_achievements.append('first_assessment')

        # Perfect score
        if result.percentage == 100:
            new_achievements.append('perfect_score')

        # High performance
        if result.percentage >= 90:
            new_achievements.append('high_performer')

        # Consistent performance
        if profile['assessments_completed'] >= 5:
            recent_scores = [r['percentage'] for r in self.progress_records[user_id][-5:]]
            if all(score >= 80 for score in recent_scores):
                new_achievements.append('consistent_learner')

        # Skill mastery
        for skill, data in profile['skills'].items():
            if data['level'] == 5 and f'master_{skill}' not in profile['achievements']:
                new_achievements.append(f'master_{skill}')

        # Add new achievements
        for achievement in new_achievements:
            if achievement not in profile['achievements']:
                profile['achievements'].append(achievement)

    def _question_to_dict(self, question: Question) -> Dict:
        """Convert Question object to dictionary for JSON serialization"""
        return {
            'id': question.id,
            'question_type': question.question_type.value,
            'title': question.title,
            'description': question.description,
            'difficulty': question.difficulty.value,
            'section_id': question.section_id,
            'subsection': question.subsection,
            'content': question.content,
            'correct_answer': question.correct_answer,
            'points': question.points,
            'time_limit': question.time_limit,
            'hints': question.hints,
            'explanation': question.explanation,
            'tags': question.tags
        }

    def _dict_to_question(self, data: Dict) -> Question:
        """Convert dictionary to Question object"""
        return Question(
            id=data['id'],
            question_type=QuestionType(data['question_type']),
            title=data['title'],
            description=data['description'],
            difficulty=DifficultyLevel(data['difficulty']),
            section_id=data['section_id'],
            subsection=data['subsection'],
            content=data['content'],
            correct_answer=data['correct_answer'],
            points=data['points'],
            time_limit=data.get('time_limit'),
            hints=data.get('hints', []),
            explanation=data.get('explanation', ''),
            tags=data.get('tags', [])
        )

    def get_user_progress(self, user_id: str) -> Dict:
        """Get comprehensive user progress report"""
        if user_id not in self.user_profiles:
            return {'error': 'User not found'}

        profile = self.user_profiles[user_id]
        progress_data = self.progress_records.get(user_id, [])

        # Calculate section-wise performance
        section_performance = {}
        for result in progress_data:
            assessment = self.assessments[result['assessment_id']]
            section_id = assessment['section_id']

            if section_id not in section_performance:
                section_performance[section_id] = []
            section_performance[section_id].append(result['percentage'])

        # Calculate averages
        section_averages = {section: sum(scores) / len(scores)
                           for section, scores in section_performance.items()}

        return {
            'user_id': user_id,
            'profile': profile,
            'section_performance': section_averages,
            'total_assessments': len(progress_data),
            'average_score': sum(r['percentage'] for r in progress_data) / len(progress_data) if progress_data else 0,
            'recent_progress': progress_data[-10:] if progress_data else []
        }

    def generate_leaderboard(self, section_id: str = None, limit: int = 10) -> List[Dict]:
        """Generate leaderboard rankings"""
        leaderboard = []

        for user_id, profile in self.user_profiles.items():
            if section_id:
                # Filter by section-specific performance
                user_progress = self.progress_records.get(user_id, [])
                section_scores = [r['percentage'] for r in user_progress
                                if self.assessments[r['assessment_id']]['section_id'] == section_id]
                avg_score = sum(section_scores) / len(section_scores) if section_scores else 0
            else:
                # Overall performance
                avg_score = profile.get('total_score', 0) / max(profile.get('assessments_completed', 1), 1)

            leaderboard.append({
                'user_id': user_id,
                'score': avg_score,
                'level': profile.get('level', 1),
                'assessments_completed': profile.get('assessments_completed', 0),
                'achievements_count': len(profile.get('achievements', []))
            })

        # Sort by score and limit results
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        return leaderboard[:limit]

if __name__ == "__main__":
    # Example usage
    engine = AssessmentEngine()

    # Create sample questions
    q1 = engine.create_question(
        question_type="multiple_choice",
        title="Linear Algebra Basics",
        description="What is the determinant of a 2x2 matrix [[a,b],[c,d]]?",
        difficulty=1,
        section_id="01_Foundational_Machine_Learning",
        subsection="linear_algebra",
        content={
            "options": ["ad-bc", "ad+bc", "ac-bd", "ac+bd"]
        },
        correct_answer="ad-bc",
        points=10,
        explanation="The determinant of a 2x2 matrix [[a,b],[c,d]] is calculated as ad-bc",
        tags=["linear_algebra", "matrix_operations"]
    )

    q2 = engine.create_question(
        question_type="coding_challenge",
        title="Linear Regression Implementation",
        description="Implement a simple linear regression algorithm from scratch",
        difficulty=2,
        section_id="01_Foundational_Machine_Learning",
        subsection="ml_algorithms",
        content={
            "description": "Write a Python function that implements linear regression using gradient descent",
            "test_cases": [
                {"input": "sample_data_1", "expected": "mse < 0.1"},
                {"input": "sample_data_2", "expected": "convergence in 1000 iterations"}
            ]
        },
        correct_answer="def linear_regression(X, y):...",
        points=25,
        tags=["ml_algorithms", "regression", "gradient_descent"]
    )

    print(f"Created {len(engine.questions)} sample questions")
    print(f"Question 1: {q1.title}")
    print(f"Question 2: {q2.title}")