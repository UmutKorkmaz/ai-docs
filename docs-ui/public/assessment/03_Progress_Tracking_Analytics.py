#!/usr/bin/env python3
"""
Progress Tracking and Analytics System
Comprehensive user progress monitoring, analytics, and visualization
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

@dataclass
class SkillMetric:
    """Individual skill performance metric"""
    skill_name: str
    current_level: float
    improvement_rate: float
    practice_sessions: int
    last_practiced: datetime
    confidence_score: float
    weakness_areas: List[str]

@dataclass
class LearningPath:
    """Personalized learning path recommendation"""
    current_section: str
    recommended_next: List[str]
    priority_topics: List[str]
    estimated_completion_time: timedelta
    difficulty_adjustment: str
    learning_style_recommendations: List[str]

@dataclass
class PerformanceAnalytics:
    """Comprehensive performance analytics"""
    user_id: str
    overall_progress: float
    skill_matrix: Dict[str, SkillMetric]
    learning_velocity: float
    knowledge_retention: float
    learning_efficiency: float
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    risk_factors: List[str]
    engagement_metrics: Dict[str, float]

class ProgressTracker:
    """Main progress tracking and analytics engine"""

    def __init__(self, data_path: str = "assessment_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        self.user_data = {}
        self.skill_definitions = self._load_skill_definitions()
        self.analytics_cache = {}

        self.load_data()

    def _load_skill_definitions(self) -> Dict[str, Dict]:
        """Load skill definitions for all sections"""
        return {
            "01_Foundational_Machine_Learning": {
                "linear_algebra": {"description": "Matrix operations, eigenvalues, SVD", "category": "mathematical"},
                "calculus": {"description": "Optimization, gradients, derivatives", "category": "mathematical"},
                "probability": {"description": "Probability theory, distributions", "category": "mathematical"},
                "statistics": {"description": "Statistical methods, hypothesis testing", "category": "mathematical"},
                "supervised_learning": {"description": "Classification, regression algorithms", "category": "algorithms"},
                "unsupervised_learning": {"description": "Clustering, dimensionality reduction", "category": "algorithms"},
                "ensemble_methods": {"description": "Random forests, boosting, stacking", "category": "algorithms"},
                "bayesian_methods": {"description": "Bayesian inference, probabilistic models", "category": "theoretical"},
                "causal_inference": {"description": "Causal relationships, counterfactuals", "category": "theoretical"}
            },
            "02_Advanced_Deep_Learning": {
                "neural_networks": {"description": "MLP, activation functions, backpropagation", "category": "foundational"},
                "convolutional_networks": {"description": "CNNs, feature extraction, architectures", "category": "architectures"},
                "recurrent_networks": {"description": "RNNs, LSTMs, GRUs, sequence modeling", "category": "architectures"},
                "transformers": {"description": "Attention mechanisms, transformer models", "category": "architectures"},
                "optimization": {"description": "Advanced optimization techniques", "category": "techniques"},
                "regularization": {"description": "Dropout, batch norm, weight decay", "category": "techniques"},
                "architecture_design": {"description": "Neural architecture design principles", "category": "design"}
            },
            "03_Natural_Language_Processing": {
                "text_processing": {"description": "Tokenization, stemming, lemmatization", "category": "foundational"},
                "language_models": {"description": "LM architectures, pre-training", "category": "models"},
                "transformers_nlp": {"description": "BERT, GPT, transformer applications", "category": "models"},
                "information_extraction": {"description": "NER, relation extraction, parsing", "category": "applications"},
                "text_generation": {"description": "Language generation, translation", "category": "applications"},
                "sentiment_analysis": {"description": "Emotion detection, opinion mining", "category": "applications"}
            },
            "04_Computer_Vision": {
                "image_processing": {"description": "Filtering, enhancement, features", "category": "foundational"},
                "object_detection": {"description": "YOLO, R-CNN, detection systems", "category": "models"},
                "image_segmentation": {"description": "Semantic, instance segmentation", "category": "models"},
                "3d_vision": {"description": "Depth estimation, 3D reconstruction", "category": "advanced"},
                "video_analysis": {"description": "Action recognition, tracking", "category": "advanced"}
            },
            "05_Generative_AI": {
                "generative_models": {"description": "VAEs, GANs, flow models", "category": "foundational"},
                "diffusion_models": {"description": "Score-based models, latent diffusion", "category": "models"},
                "large_language_models": {"description": "LLM architectures, fine-tuning", "category": "models"},
                "multimodal_generation": {"description": "Text-to-image, cross-modal", "category": "advanced"},
                "creative_ai": {"description": "Art generation, music, writing", "category": "applications"}
            }
        }

    def load_data(self):
        """Load user progress data"""
        user_data_file = self.data_path / "user_progress.json"
        if user_data_file.exists():
            with open(user_data_file, 'r') as f:
                self.user_data = json.load(f)

    def save_data(self):
        """Save user progress data"""
        user_data_file = self.data_path / "user_progress.json"
        with open(user_data_file, 'w') as f:
            json.dump(self.user_data, f, indent=2, default=str)

    def record_assessment_result(self, user_id: str, result: Dict):
        """Record assessment result and update progress metrics"""
        if user_id not in self.user_data:
            self._initialize_user_profile(user_id)

        user_profile = self.user_data[user_id]

        # Add assessment record
        if 'assessment_history' not in user_profile:
            user_profile['assessment_history'] = []

        user_profile['assessment_history'].append({
            'timestamp': datetime.now().isoformat(),
            'assessment_id': result['assessment_id'],
            'section_id': result.get('section_id', ''),
            'score': result['percentage'],
            'total_points': result['total_score'],
            'max_points': result['max_score'],
            'time_taken': result['time_taken'],
            'skill_gains': result.get('skill_gains', {})
        })

        # Update skill metrics
        self._update_skill_metrics(user_id, result)

        # Update learning path
        self._update_learning_path(user_id)

        # Clear analytics cache for this user
        if user_id in self.analytics_cache:
            del self.analytics_cache[user_id]

        self.save_data()

    def _initialize_user_profile(self, user_id: str):
        """Initialize a new user profile"""
        self.user_data[user_id] = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'current_level': 1,
            'experience_points': 0,
            'skills': {},
            'learning_path': {},
            'assessment_history': [],
            'achievements': [],
            'preferences': {
                'learning_style': 'visual',  # visual, auditory, kinesthetic
                'difficulty_preference': 'adaptive',
                'session_length': 30  # minutes
            }
        }

        # Initialize skills for all sections
        for section_id, skills in self.skill_definitions.items():
            for skill_name in skills:
                self.user_data[user_id]['skills'][f"{section_id}_{skill_name}"] = {
                    'level': 0.0,
                    'practice_count': 0,
                    'last_practiced': None,
                    'improvement_rate': 0.0,
                    'confidence_score': 0.0
                }

    def _update_skill_metrics(self, user_id: str, result: Dict):
        """Update individual skill metrics based on assessment results"""
        user_profile = self.user_data[user_id]
        section_id = result.get('section_id', '')

        # Update skills based on assessment performance
        for skill_name, gain in result.get('skill_gains', {}).items():
            full_skill_name = f"{section_id}_{skill_name}"

            if full_skill_name in user_profile['skills']:
                skill_data = user_profile['skills'][full_skill_name]
                old_level = skill_data['level']
                practice_count = skill_data['practice_count']

                # Calculate new level with diminishing returns
                learning_rate = 0.1 / (1 + practice_count * 0.05)  # Diminishing returns
                new_level = old_level + (gain / 100.0) * learning_rate
                skill_data['level'] = min(100.0, new_level)

                # Update improvement rate
                if practice_count > 0:
                    skill_data['improvement_rate'] = (skill_data['level'] - old_level) / practice_count
                else:
                    skill_data['improvement_rate'] = skill_data['level']

                # Update other metrics
                skill_data['practice_count'] += 1
                skill_data['last_practiced'] = datetime.now().isoformat()
                skill_data['confidence_score'] = min(100.0, skill_data['confidence_score'] + gain * 0.1)

    def _update_learning_path(self, user_id: str):
        """Update personalized learning path based on current progress"""
        user_profile = self.user_data[user_id]
        current_skills = user_profile['skills']

        # Analyze current performance and identify gaps
        skill_levels = {skill: data['level'] for skill, data in current_skills.items()}
        weak_skills = [skill for skill, level in skill_levels.items() if level < 50]
        strong_skills = [skill for skill, level in skill_levels.items() if level >= 80]

        # Determine current section based on completed assessments
        completed_sections = set()
        for assessment in user_profile.get('assessment_history', []):
            section_id = assessment.get('section_id', '')
            if section_id and assessment['score'] >= 70:
                completed_sections.add(section_id)

        # Generate learning path recommendations
        learning_path = {
            'current_focus': self._determine_current_section(completed_sections),
            'priority_skills': weak_skills[:5],  # Top 5 weakest skills
            'recommended_sections': self._get_next_sections(completed_sections),
            'difficulty_adjustment': self._get_difficulty_adjustment(user_id),
            'estimated_timeline': self._estimate_completion_timeline(user_id),
            'milestones': self._generate_milestones(user_id)
        }

        user_profile['learning_path'] = learning_path

    def _determine_current_section(self, completed_sections: set) -> str:
        """Determine which section the user should focus on"""
        section_order = [
            "01_Foundational_Machine_Learning",
            "02_Advanced_Deep_Learning",
            "03_Natural_Language_Processing",
            "04_Computer_Vision",
            "05_Generative_AI"
        ]

        for section in section_order:
            if section not in completed_sections:
                return section

        return "01_Foundational_Machine_Learning"  # Default to review

    def _get_next_sections(self, completed_sections: set) -> List[str]:
        """Get recommended next sections based on prerequisites"""
        section_order = [
            "01_Foundational_Machine_Learning",
            "02_Advanced_Deep_Learning",
            "03_Natural_Language_Processing",
            "04_Computer_Vision",
            "05_Generative_AI"
        ]

        next_sections = []
        for section in section_order:
            if section not in completed_sections:
                next_sections.append(section)
                if len(next_sections) >= 3:  # Recommend next 3 sections
                    break

        return next_sections

    def _get_difficulty_adjustment(self, user_id: str) -> str:
        """Determine if difficulty should be increased, decreased, or maintained"""
        user_profile = self.user_data[user_id]
        recent_assessments = user_profile.get('assessment_history', [])[-5:]  # Last 5 assessments

        if len(recent_assessments) < 3:
            return "maintain"

        recent_scores = [a['score'] for a in recent_assessments]
        avg_score = sum(recent_scores) / len(recent_scores)

        if avg_score >= 90:
            return "increase"
        elif avg_score <= 60:
            return "decrease"
        else:
            return "maintain"

    def _estimate_completion_timeline(self, user_id: str) -> Dict:
        """Estimate time to complete current learning goals"""
        user_profile = self.user_data[user_id]
        current_skills = user_profile['skills']

        # Calculate average skill levels by section
        section_progress = {}
        for skill_id, skill_data in current_skills.items():
            section_id = skill_id.split('_')[0] + '_' + skill_id.split('_')[1]
            if section_id not in section_progress:
                section_progress[section_id] = []
            section_progress[section_id].append(skill_data['level'])

        avg_progress = {section: sum(levels) / len(levels)
                       for section, levels in section_progress.items()}

        # Estimate remaining time (rough heuristic)
        remaining_sections = len([s for s in avg_progress.keys() if avg_progress[s] < 80])
        estimated_hours = remaining_sections * 20  # 20 hours per section average

        return {
            'estimated_hours': estimated_hours,
            'completion_date': (datetime.now() + timedelta(hours=estimated_hours)).isoformat(),
            'remaining_sections': remaining_sections
        }

    def _generate_milestones(self, user_id: str) -> List[Dict]:
        """Generate learning milestones for the user"""
        user_profile = self.user_data[user_id]
        current_level = user_profile.get('current_level', 1)

        milestones = []

        # Level-based milestones
        for level in range(current_level + 1, min(current_level + 6, 31)):
            exp_needed = level * 1000
            milestones.append({
                'type': 'level',
                'target': level,
                'description': f'Reach Level {level}',
                'exp_required': exp_needed,
                'estimated_completion': f"Level {level}"
            })

        # Skill-based milestones
        weak_skills = [(skill, data) for skill, data in user_profile['skills'].items()
                      if data['level'] < 50]
        for skill, data in weak_skills[:3]:
            milestones.append({
                'type': 'skill_mastery',
                'target': skill,
                'description': f'Master {skill}',
                'current_level': data['level'],
                'target_level': 80
            })

        return milestones

    def get_comprehensive_analytics(self, user_id: str) -> PerformanceAnalytics:
        """Generate comprehensive performance analytics for a user"""

        if user_id not in self.user_data:
            raise ValueError(f"User {user_id} not found")

        # Check cache first
        if user_id in self.analytics_cache:
            cache_time = self.analytics_cache[user_id]['timestamp']
            if (datetime.now() - cache_time).total_seconds() < 3600:  # 1 hour cache
                return self.analytics_cache[user_id]['analytics']

        user_profile = self.user_data[user_id]

        # Calculate overall progress
        overall_progress = self._calculate_overall_progress(user_id)

        # Generate skill matrix
        skill_matrix = self._generate_skill_matrix(user_id)

        # Calculate learning metrics
        learning_velocity = self._calculate_learning_velocity(user_id)
        knowledge_retention = self._calculate_knowledge_retention(user_id)
        learning_efficiency = self._calculate_learning_efficiency(user_id)

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(user_id)

        # Generate recommendations
        recommendations = self._generate_recommendations(user_id)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(user_id)

        # Calculate engagement metrics
        engagement_metrics = self._calculate_engagement_metrics(user_id)

        analytics = PerformanceAnalytics(
            user_id=user_id,
            overall_progress=overall_progress,
            skill_matrix=skill_matrix,
            learning_velocity=learning_velocity,
            knowledge_retention=knowledge_retention,
            learning_efficiency=learning_efficiency,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            risk_factors=risk_factors,
            engagement_metrics=engagement_metrics
        )

        # Cache the result
        self.analytics_cache[user_id] = {
            'timestamp': datetime.now(),
            'analytics': analytics
        }

        return analytics

    def _calculate_overall_progress(self, user_id: str) -> float:
        """Calculate overall progress percentage"""
        user_profile = self.user_data[user_id]
        skills = user_profile['skills']

        if not skills:
            return 0.0

        total_skill_level = sum(skill_data['level'] for skill_data in skills.values())
        max_possible = len(skills) * 100.0

        return (total_skill_level / max_possible) * 100.0

    def _generate_skill_matrix(self, user_id: str) -> Dict[str, SkillMetric]:
        """Generate detailed skill matrix"""
        user_profile = self.user_data[user_id]
        skill_matrix = {}

        for skill_id, skill_data in user_profile['skills'].items():
            # Calculate confidence score based on consistency
            assessment_scores = []
            for assessment in user_profile.get('assessment_history', []):
                if skill_id in assessment.get('skill_gains', {}):
                    assessment_scores.append(assessment['score'])

            confidence_score = np.std(assessment_scores) if assessment_scores else 50.0
            confidence_score = max(0, 100 - confidence_score)  # Lower std = higher confidence

            # Identify weakness areas
            weakness_areas = []
            if skill_data['improvement_rate'] < 0.5:
                weakness_areas.append("slow_progress")
            if confidence_score < 70:
                weakness_areas.append("inconsistent_performance")
            if skill_data['practice_count'] < 3:
                weakness_areas.append("insufficient_practice")

            skill_metric = SkillMetric(
                skill_name=skill_id,
                current_level=skill_data['level'],
                improvement_rate=skill_data['improvement_rate'],
                practice_sessions=skill_data['practice_count'],
                last_practiced=datetime.fromisoformat(skill_data['last_practiced']) if skill_data['last_practiced'] else datetime.now(),
                confidence_score=confidence_score,
                weakness_areas=weakness_areas
            )

            skill_matrix[skill_id] = skill_metric

        return skill_matrix

    def _calculate_learning_velocity(self, user_id: str) -> float:
        """Calculate how quickly the user is learning"""
        user_profile = self.user_data[user_id]
        assessments = user_profile.get('assessment_history', [])

        if len(assessments) < 2:
            return 0.0

        # Calculate progress rate over time
        recent_assessments = assessments[-10:]  # Last 10 assessments
        if len(recent_assessments) < 2:
            return 0.0

        time_span = (datetime.fromisoformat(recent_assessments[-1]['timestamp']) -
                    datetime.fromisoformat(recent_assessments[0]['timestamp'])).total_seconds()

        if time_span == 0:
            return 0.0

        progress_gain = sum(a['score'] for a in recent_assessments) / len(recent_assessments)
        velocity = progress_gain / (time_span / 3600)  # Progress per hour

        return velocity

    def _calculate_knowledge_retention(self, user_id: str) -> float:
        """Calculate knowledge retention based on spaced repetition"""
        user_profile = self.user_data[user_id]
        skills = user_profile['skills']

        if not skills:
            return 0.0

        retention_scores = []
        current_time = datetime.now()

        for skill_id, skill_data in skills.items():
            if skill_data['last_practiced']:
                last_practiced = datetime.fromisoformat(skill_data['last_practiced'])
                days_since_practice = (current_time - last_practiced).days

                # Forgetting curve calculation
                retention_rate = skill_data['level'] * np.exp(-days_since_practice / 30.0)  # 30-day half-life
                retention_scores.append(retention_rate)

        return np.mean(retention_scores) if retention_scores else 0.0

    def _calculate_learning_efficiency(self, user_id: str) -> float:
        """Calculate learning efficiency (progress vs time invested)"""
        user_profile = self.user_data[user_id]
        assessments = user_profile.get('assessment_history', [])

        if not assessments:
            return 0.0

        total_time = sum(a['time_taken'] for a in assessments) / 3600.0  # Convert to hours
        total_progress = sum(a['score'] for a in assessments)

        if total_time == 0:
            return 0.0

        efficiency = total_progress / total_time
        return efficiency

    def _identify_strengths_weaknesses(self, user_id: str) -> Tuple[List[str], List[str]]:
        """Identify user's strengths and weaknesses"""
        user_profile = self.user_data[user_id]
        skills = user_profile['skills']

        strength_threshold = 80.0
        weakness_threshold = 50.0

        strengths = [skill_id for skill_id, skill_data in skills.items()
                    if skill_data['level'] >= strength_threshold]

        weaknesses = [skill_id for skill_id, skill_data in skills.items()
                     if skill_data['level'] < weakness_threshold]

        return strengths, weaknesses

    def _generate_recommendations(self, user_id: str) -> List[str]:
        """Generate personalized learning recommendations"""
        user_profile = self.user_data[user_id]
        recommendations = []

        # Analyze recent performance
        recent_assessments = user_profile.get('assessment_history', [])[-5:]
        if recent_assessments:
            avg_score = sum(a['score'] for a in recent_assessments) / len(recent_assessments)

            if avg_score < 60:
                recommendations.append("Focus on foundational concepts before advancing")
            elif avg_score > 90:
                recommendations.append("Consider more challenging material to maintain growth")

        # Analyze practice patterns
        skills_with_low_practice = [skill_id for skill_id, skill_data in user_profile['skills'].items()
                                  if skill_data['practice_count'] < 3]
        if skills_with_low_practice:
            recommendations.append(f"Increase practice in under-practiced areas: {skills_with_low_practice[:3]}")

        # Analyze improvement rates
        slow_progress_skills = [skill_id for skill_id, skill_data in user_profile['skills'].items()
                               if skill_data['improvement_rate'] < 0.3 and skill_data['practice_count'] > 0]
        if slow_progress_skills:
            recommendations.append(f"Review learning approach for: {slow_progress_skills[:2]}")

        # Time-based recommendations
        if len(recent_assessments) < 2:
            recommendations.append("Establish regular practice schedule for consistent progress")

        return recommendations

    def _identify_risk_factors(self, user_id: str) -> List[str]:
        """Identify potential risk factors for learning progress"""
        user_profile = self.user_data[user_id]
        risk_factors = []

        # Check for inconsistent practice
        assessments = user_profile.get('assessment_history', [])
        if len(assessments) > 5:
            timestamps = [datetime.fromisoformat(a['timestamp']) for a in assessments[-10:]]
            time_diffs = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]

            if np.std(time_diffs) > 7:  # High variance in practice intervals
                risk_factors.append("inconsistent_practice_schedule")

        # Check for declining performance
        if len(assessments) > 3:
            recent_scores = [a['score'] for a in assessments[-3:]]
            earlier_scores = [a['score'] for a in assessments[-6:-3]]

            if np.mean(recent_scores) < np.mean(earlier_scores) - 10:
                risk_factors.append("declining_performance")

        # Check for knowledge gaps
        weak_skills = len([s for s in user_profile['skills'].values() if s['level'] < 40])
        if weak_skills > len(user_profile['skills']) * 0.3:
            risk_factors.append("significant_knowledge_gaps")

        return risk_factors

    def _calculate_engagement_metrics(self, user_id: str) -> Dict[str, float]:
        """Calculate user engagement metrics"""
        user_profile = self.user_data[user_id]
        assessments = user_profile.get('assessment_history', [])

        if not assessments:
            return {
                'session_frequency': 0.0,
                'average_session_length': 0.0,
                'engagement_score': 0.0,
                'consistency_index': 0.0
            }

        # Session frequency (sessions per week)
        if len(assessments) > 1:
            time_span = (datetime.fromisoformat(assessments[-1]['timestamp']) -
                        datetime.fromisoformat(assessments[0]['timestamp'])).days
            session_frequency = (len(assessments) / time_span) * 7 if time_span > 0 else 0
        else:
            session_frequency = 0

        # Average session length
        avg_session_length = np.mean([a['time_taken'] / 60.0 for a in assessments])  # Convert to minutes

        # Engagement score (composite metric)
        completion_rate = sum(1 for a in assessments if a['score'] >= 70) / len(assessments)
        engagement_score = (session_frequency * 0.3 + avg_session_length * 0.1 + completion_rate * 0.6) * 100

        # Consistency index
        if len(assessments) > 5:
            timestamps = [datetime.fromisoformat(a['timestamp']) for a in assessments[-10:]]
            time_diffs = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
            consistency_index = 100 / (1 + np.std(time_diffs)) if time_diffs else 0
        else:
            consistency_index = 0

        return {
            'session_frequency': session_frequency,
            'average_session_length': avg_session_length,
            'engagement_score': engagement_score,
            'consistency_index': consistency_index
        }

    def generate_progress_visualization(self, user_id: str) -> Dict[str, str]:
        """Generate visualization charts for user progress"""
        if user_id not in self.user_data:
            return {}

        user_profile = self.user_data[user_id]
        assessments = user_profile.get('assessment_history', [])

        # Create progress over time chart
        if assessments:
            df = pd.DataFrame(assessments)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            fig_progress = px.line(df, x='timestamp', y='score',
                                  title='Assessment Performance Over Time',
                                  labels={'score': 'Score (%)', 'timestamp': 'Date'})
            fig_progress.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Passing Threshold")
            progress_chart = fig_progress.to_html(include_plotlyjs='cdn')
        else:
            progress_chart = ""

        # Create skill radar chart
        skills_data = []
        for skill_id, skill_data in user_profile['skills'].items():
            skills_data.append({
                'skill': skill_id.split('_')[-1],  # Use last part of skill name
                'level': skill_data['level'],
                'category': skill_id.split('_')[0]
            })

        if skills_data:
            df_skills = pd.DataFrame(skills_data)
            fig_skills = px.line_polar(df_skills, r='level', theta='skill',
                                       line_close=True, title='Current Skill Levels')
            skill_chart = fig_skills.to_html(include_plotlyjs='cdn')
        else:
            skill_chart = ""

        # Create learning velocity chart
        if len(assessments) > 1:
            df['rolling_avg'] = df['score'].rolling(window=min(3, len(df))).mean()
            fig_velocity = px.line(df, x='timestamp', y='rolling_avg',
                                  title='Learning Velocity (Rolling Average)',
                                  labels={'rolling_avg': 'Average Score (%)', 'timestamp': 'Date'})
            velocity_chart = fig_velocity.to_html(include_plotlyjs='cdn')
        else:
            velocity_chart = ""

        return {
            'progress_chart': progress_chart,
            'skill_chart': skill_chart,
            'velocity_chart': velocity_chart
        }

    def generate_learning_path_recommendation(self, user_id: str) -> LearningPath:
        """Generate detailed learning path recommendation"""
        if user_id not in self.user_data:
            raise ValueError(f"User {user_id} not found")

        user_profile = self.user_data[user_id]

        # Determine current section focus
        current_section = user_profile.get('learning_path', {}).get('current_focus',
                        self._determine_current_section(set()))

        # Get skill-based recommendations
        weak_skills = [skill_id for skill_id, skill_data in user_profile['skills'].items()
                      if skill_data['level'] < 60]

        # Priority topics based on prerequisites and current weaknesses
        priority_topics = self._get_priority_topics(current_section, weak_skills)

        # Learning style recommendations
        learning_style = user_profile.get('preferences', {}).get('learning_style', 'visual')
        style_recommendations = self._get_learning_style_recommendations(learning_style)

        # Estimate completion time
        estimated_time = timedelta(hours=len(priority_topics) * 2)  # 2 hours per topic

        # Difficulty adjustment
        difficulty_adj = user_profile.get('learning_path', {}).get('difficulty_adjustment', 'maintain')

        return LearningPath(
            current_section=current_section,
            recommended_next=self._get_next_sections(set()),
            priority_topics=priority_topics,
            estimated_completion_time=estimated_time,
            difficulty_adjustment=difficulty_adj,
            learning_style_recommendations=style_recommendations
        )

    def _get_priority_topics(self, current_section: str, weak_skills: List[str]) -> List[str]:
        """Get priority learning topics based on current section and weaknesses"""
        section_weaknesses = [skill for skill in weak_skills if current_section in skill]

        # Map skills to specific topics
        topic_mapping = {
            'linear_algebra': ['Matrix operations', 'Eigenvalue decomposition', 'SVD'],
            'calculus': ['Gradient descent', 'Optimization techniques', 'Partial derivatives'],
            'probability': ['Bayes theorem', 'Probability distributions', 'Conditional probability'],
            'supervised_learning': ['Classification algorithms', 'Regression techniques', 'Model evaluation'],
            'neural_networks': ['Backpropagation', 'Activation functions', 'Network architectures'],
            'transformers': ['Attention mechanisms', 'Self-attention', 'Transformer architecture']
        }

        priority_topics = []
        for weakness in section_weaknesses:
            skill_name = weakness.split('_')[-1]
            if skill_name in topic_mapping:
                priority_topics.extend(topic_mapping[skill_name])

        return priority_topics[:10]  # Return top 10 priority topics

    def _get_learning_style_recommendations(self, learning_style: str) -> List[str]:
        """Get recommendations based on learning style"""
        recommendations = {
            'visual': [
                'Use diagrams and visualizations',
                'Watch video tutorials',
                'Create mind maps of concepts',
                'Use color-coded notes'
            ],
            'auditory': [
                'Listen to lecture recordings',
                'Participate in discussions',
                'Use verbal repetition',
                'Explain concepts aloud'
            ],
            'kinesthetic': [
                'Hands-on coding exercises',
                'Interactive tutorials',
                'Real-world projects',
                'Physical note-taking'
            ]
        }

        return recommendations.get(learning_style, recommendations['visual'])

    def get_leaderboard_data(self, timeframe: str = 'all_time') -> pd.DataFrame:
        """Generate leaderboard data for all users"""
        leaderboard_data = []

        for user_id, user_profile in self.user_data.items():
            # Calculate metrics for ranking
            overall_progress = self._calculate_overall_progress(user_id)
            current_level = user_profile.get('current_level', 1)
            total_assessments = len(user_profile.get('assessment_history', []))
            achievements_count = len(user_profile.get('achievements', []))

            # Filter by timeframe
            if timeframe == 'weekly':
                cutoff_date = datetime.now() - timedelta(days=7)
                recent_assessments = [a for a in user_profile.get('assessment_history', [])
                                    if datetime.fromisoformat(a['timestamp']) > cutoff_date]
                weekly_score = sum(a['score'] for a in recent_assessments) / len(recent_assessments) if recent_assessments else 0
            elif timeframe == 'monthly':
                cutoff_date = datetime.now() - timedelta(days=30)
                recent_assessments = [a for a in user_profile.get('assessment_history', [])
                                    if datetime.fromisoformat(a['timestamp']) > cutoff_date]
                monthly_score = sum(a['score'] for a in recent_assessments) / len(recent_assessments) if recent_assessments else 0
            else:  # all_time
                weekly_score = monthly_score = 0

            leaderboard_data.append({
                'user_id': user_id,
                'overall_progress': overall_progress,
                'current_level': current_level,
                'total_assessments': total_assessments,
                'achievements_count': achievements_count,
                'weekly_score': weekly_score,
                'monthly_score': monthly_score
            })

        df = pd.DataFrame(leaderboard_data)

        # Calculate composite ranking score
        df['ranking_score'] = (
            df['overall_progress'] * 0.4 +
            df['current_level'] * 0.2 +
            df['achievements_count'] * 0.2 +
            df['weekly_score'] * 0.1 +
            df['monthly_score'] * 0.1
        )

        return df.sort_values('ranking_score', ascending=False)

if __name__ == "__main__":
    # Example usage
    tracker = ProgressTracker()

    # Simulate some user data
    sample_result = {
        'assessment_id': 'test_001',
        'section_id': '01_Foundational_Machine_Learning',
        'percentage': 85.0,
        'total_score': 85,
        'max_score': 100,
        'time_taken': 1800,
        'skill_gains': {
            'linear_algebra': 80,
            'calculus': 75,
            'probability': 90
        }
    }

    # Record sample assessment
    tracker.record_assessment_result('user_001', sample_result)

    # Get analytics
    try:
        analytics = tracker.get_comprehensive_analytics('user_001')
        print(f"Overall Progress: {analytics.overall_progress:.1f}%")
        print(f"Learning Velocity: {analytics.learning_velocity:.2f}")
        print(f"Knowledge Retention: {analytics.knowledge_retention:.1f}%")
        print(f"Strengths: {analytics.strengths}")
        print(f"Weaknesses: {analytics.weaknesses}")
        print(f"Recommendations: {analytics.recommendations}")
    except ValueError as e:
        print(f"Error: {e}")

    # Generate learning path
    try:
        learning_path = tracker.generate_learning_path_recommendation('user_001')
        print(f"\nLearning Path:")
        print(f"Current Section: {learning_path.current_section}")
        print(f"Priority Topics: {learning_path.priority_topics[:3]}")
        print(f"Estimated Time: {learning_path.estimated_completion_time}")
    except ValueError as e:
        print(f"Error: {e}")

    # Generate leaderboard
    leaderboard = tracker.get_leaderboard_data()
    print(f"\nLeaderboard (Top 5):")
    print(leaderboard[['user_id', 'ranking_score', 'overall_progress']].head())

    print("\nProgress tracking system initialized successfully!")