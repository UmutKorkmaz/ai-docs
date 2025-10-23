#!/usr/bin/env python3
"""
Achievement and Certification System
Comprehensive badge, certificate, and credential management
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import uuid
import base64

class AchievementType(Enum):
    SKILL_MASTERY = "skill_mastery"
    SECTION_COMPLETION = "section_completion"
    PERFORMANCE = "performance"
    CONSISTENCY = "consistency"
    INNOVATION = "innovation"
    COLLABORATION = "collaboration"
    COMMUNITY = "community"
    SPECIAL = "special"

class CertificateType(Enum):
    SECTION_CERTIFICATE = "section_certificate"
    SKILL_CERTIFICATE = "skill_certificate"
    PROFESSIONAL_CERTIFICATE = "professional_certificate"
    RESEARCH_CERTIFICATE = "research_certificate"
    MASTERY_CERTIFICATE = "mastery_certificate"

class BadgeTier(Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"

@dataclass
class Achievement:
    """Individual achievement/badge definition"""
    id: str
    name: str
    description: str
    type: AchievementType
    tier: BadgeTier
    icon_url: str
    requirements: Dict[str, Any]
    points_value: int
    rarity: float  # 0.0 to 1.0, lower = rarer
    category: str
    expiration_days: Optional[int] = None
    prerequisites: List[str] = None

    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []

@dataclass
class UserAchievement:
    """User's earned achievement record"""
    user_id: str
    achievement_id: str
    earned_at: datetime
    expires_at: Optional[datetime] = None
    progress: Dict[str, float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.progress is None:
            self.progress = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Certificate:
    """Certificate definition"""
    id: str
    name: str
    type: CertificateType
    description: str
    requirements: Dict[str, Any]
    template_id: str
    validity_days: Optional[int] = None
    issuer: str = "AI Documentation Project"
    skills_certified: List[str] = None
    version: str = "1.0"

    def __post_init__(self):
        if self.skills_certified is None:
            self.skills_certified = []

@dataclass
class UserCertificate:
    """User's earned certificate"""
    user_id: str
    certificate_id: str
    issued_at: datetime
    expires_at: Optional[datetime] = None
    verification_code: str
    credential_id: str
    status: str = "active"  # active, expired, revoked
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class LeaderboardEntry:
    """Leaderboard entry structure"""
    user_id: str
    display_name: str
    score: float
    rank: int
    achievements_count: int
    certificates_count: int
    level: int
    special_badges: List[str] = None

    def __post_init__(self):
        if self.special_badges is None:
            self.special_badges = []

class AchievementSystem:
    """Main achievement and certification system"""

    def __init__(self, data_path: str = "assessment_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        self.achievements = {}
        self.certificates = {}
        self.user_achievements = {}
        self.user_certificates = {}
        self.leaderboard_cache = {}

        self.load_data()
        self._initialize_default_achievements()
        self._initialize_default_certificates()

    def load_data(self):
        """Load achievement and certificate data"""
        # Load achievements
        achievements_file = self.data_path / "achievements.json"
        if achievements_file.exists():
            with open(achievements_file, 'r') as f:
                achievements_data = json.load(f)
                for aid, adata in achievements_data.items():
                    self.achievements[aid] = self._dict_to_achievement(adata)

        # Load certificates
        certificates_file = self.data_path / "certificates.json"
        if certificates_file.exists():
            with open(certificates_file, 'r') as f:
                certificates_data = json.load(f)
                for cid, cdata in certificates_data.items():
                    self.certificates[cid] = self._dict_to_certificate(cdata)

        # Load user achievements
        user_achievements_file = self.data_path / "user_achievements.json"
        if user_achievements_file.exists():
            with open(user_achievements_file, 'r') as f:
                self.user_achievements = json.load(f)

        # Load user certificates
        user_certificates_file = self.data_path / "user_certificates.json"
        if user_certificates_file.exists():
            with open(user_certificates_file, 'r') as f:
                self.user_certificates = json.load(f)

    def save_data(self):
        """Save achievement and certificate data"""
        # Save achievements
        achievements_file = self.data_path / "achievements.json"
        achievements_data = {aid: self._achievement_to_dict(a) for aid, a in self.achievements.items()}
        with open(achievements_file, 'w') as f:
            json.dump(achievements_data, f, indent=2, default=str)

        # Save certificates
        certificates_file = self.data_path / "certificates.json"
        certificates_data = {cid: self._certificate_to_dict(c) for cid, c in self.certificates.items()}
        with open(certificates_file, 'w') as f:
            json.dump(certificates_data, f, indent=2, default=str)

        # Save user achievements
        user_achievements_file = self.data_path / "user_achievements.json"
        with open(user_achievements_file, 'w') as f:
            json.dump(self.user_achievements, f, indent=2, default=str)

        # Save user certificates
        user_certificates_file = self.data_path / "user_certificates.json"
        with open(user_certificates_file, 'w') as f:
            json.dump(self.user_certificates, f, indent=2, default=str)

    def _initialize_default_achievements(self):
        """Initialize default achievements if none exist"""
        if not self.achievements:
            # Skill Mastery Achievements
            skill_achievements = [
                Achievement(
                    id="first_steps",
                    name="First Steps",
                    description="Complete your first assessment",
                    type=AchievementType.PERFORMANCE,
                    tier=BadgeTier.BRONZE,
                    icon_url="/badges/first_steps.png",
                    requirements={"assessments_completed": 1},
                    points_value=50,
                    rarity=1.0,
                    category="beginner"
                ),
                Achievement(
                    id="quick_learner",
                    name="Quick Learner",
                    description="Complete 5 assessments with 90%+ average score",
                    type=AchievementType.PERFORMANCE,
                    tier=BadgeTier.SILVER,
                    icon_url="/badges/quick_learner.png",
                    requirements={
                        "assessments_completed": 5,
                        "average_score": 90,
                        "timeframe_days": 7
                    },
                    points_value=150,
                    rarity=0.7,
                    category="performance"
                ),
                Achievement(
                    id="perfectionist",
                    name="Perfectionist",
                    description="Achieve 100% score on 3 different assessments",
                    type=AchievementType.PERFORMANCE,
                    tier=BadgeTier.GOLD,
                    icon_url="/badges/perfectionist.png",
                    requirements={
                        "perfect_scores": 3,
                        "unique_assessments": 3
                    },
                    points_value=200,
                    rarity=0.3,
                    category="performance"
                ),
                Achievement(
                    id="consistent_learner",
                    name="Consistent Learner",
                    description="Complete assessments for 7 consecutive days",
                    type=AchievementType.CONSISTENCY,
                    tier=BadgeTier.SILVER,
                    icon_url="/badges/consistent_learner.png",
                    requirements={
                        "consecutive_days": 7,
                        "daily_assessments": 1
                    },
                    points_value=100,
                    rarity=0.6,
                    category="consistency"
                ),
                Achievement(
                    id="section_master",
                    name="Section Master",
                    description="Complete all assessments in a section with 80%+ average",
                    type=AchievementType.SECTION_COMPLETION,
                    tier=BadgeTier.GOLD,
                    icon_url="/badges/section_master.png",
                    requirements={
                        "section_completion": 1,
                        "section_average": 80
                    },
                    points_value=300,
                    rarity=0.4,
                    category="mastery"
                ),
                Achievement(
                    id="skill_expert",
                    name="Skill Expert",
                    description="Master a specific skill area (90%+ proficiency)",
                    type=AchievementType.SKILL_MASTERY,
                    tier=BadgeTier.PLATINUM,
                    icon_url="/badges/skill_expert.png",
                    requirements={
                        "skill_mastery": 1,
                        "skill_level": 90
                    },
                    points_value=500,
                    rarity=0.2,
                    category="expertise"
                ),
                Achievement(
                    id="innovator",
                    name="Innovator",
                    description="Create a novel solution that receives peer recognition",
                    type=AchievementType.INNOVATION,
                    tier=BadgeTier.PLATINUM,
                    icon_url="/badges/innovator.png",
                    requirements={
                        "peer_recognitions": 5,
                        "novel_solutions": 1
                    },
                    points_value=400,
                    rarity=0.1,
                    category="innovation"
                ),
                Achievement(
                    id="community_helper",
                    name="Community Helper",
                    description="Help 10 other users with their learning journey",
                    type=AchievementType.COLLABORATION,
                    tier=BadgeTier.SILVER,
                    icon_url="/badges/community_helper.png",
                    requirements={
                        "help_interactions": 10,
                        "positive_feedback": 8
                    },
                    points_value=150,
                    rarity=0.5,
                    category="community"
                ),
                Achievement(
                    id="marathon_learner",
                    name="Marathon Learner",
                    description="Complete 100 assessments total",
                    type=AchievementType.CONSISTENCY,
                    tier=BadgeTier.GOLD,
                    icon_url="/badges/marathon_learner.png",
                    requirements={
                        "total_assessments": 100
                    },
                    points_value=250,
                    rarity=0.3,
                    category="consistency"
                ),
                Achievement(
                    id="speed_demon",
                    name="Speed Demon",
                    description="Complete an assessment in under 50% of the time limit",
                    type=AchievementType.PERFORMANCE,
                    tier=BadgeTier.SILVER,
                    icon_url="/badges/speed_demon.png",
                    requirements={
                        "speed_completions": 1,
                        "time_ratio": 0.5
                    },
                    points_value=100,
                    rarity=0.6,
                    category="performance"
                )
            ]

            for achievement in skill_achievements:
                self.achievements[achievement.id] = achievement

    def _initialize_default_certificates(self):
        """Initialize default certificates if none exist"""
        if not self.certificates:
            certificates = [
                Certificate(
                    id="foundational_ml_certificate",
                    name="Foundational Machine Learning Certificate",
                    type=CertificateType.SECTION_CERTIFICATE,
                    description="Demonstrates mastery of foundational machine learning concepts",
                    requirements={
                        "section_completion": "01_Foundational_Machine_Learning",
                        "minimum_score": 80,
                        "required_assessments": 10
                    },
                    template_id="section_template",
                    validity_days=365,
                    skills_certified=[
                        "linear_algebra", "calculus", "probability", "statistics",
                        "supervised_learning", "unsupervised_learning", "ensemble_methods"
                    ],
                    version="1.0"
                ),
                Certificate(
                    id="deep_learning_certificate",
                    name="Deep Learning Practitioner Certificate",
                    type=CertificateType.PROFESSIONAL_CERTIFICATE,
                    description="Professional certification in deep learning techniques and applications",
                    requirements={
                        "section_completions": ["02_Advanced_Deep_Learning"],
                        "practical_projects": 3,
                        "minimum_score": 85,
                        "peer_review": True
                    },
                    template_id="professional_template",
                    validity_days=730,
                    skills_certified=[
                        "neural_networks", "convolutional_networks", "recurrent_networks",
                        "transformers", "optimization", "regularization"
                    ],
                    version="1.0"
                ),
                Certificate(
                    id="ai_ethics_safety_certificate",
                    name="AI Ethics and Safety Specialist",
                    type=CertificateType.SKILL_CERTIFICATE,
                    description="Specialized certification in AI ethics, safety, and responsible development",
                    requirements={
                        "section_completion": "07_AI_Ethics_and_Safety",
                        "case_studies_completed": 5,
                        "ethics_project": 1,
                        "minimum_score": 75
                    },
                    template_id="skill_template",
                    validity_days=1095,
                    skills_certified=[
                        "ethical_frameworks", "safety_research", "bias_mitigation",
                        "responsible_ai", "governance", "transparency"
                    ],
                    version="1.0"
                ),
                Certificate(
                    id="nlp_specialist_certificate",
                    name="NLP Specialist Certificate",
                    type=CertificateType.SKILL_CERTIFICATE,
                    description="Specialized certification in natural language processing and understanding",
                    requirements={
                        "section_completion": "03_Natural_Language_Processing",
                        "practical_applications": 4,
                        "model_deployment": 1,
                        "minimum_score": 80
                    },
                    template_id="skill_template",
                    validity_days=730,
                    skills_certified=[
                        "text_processing", "language_models", "transformers_nlp",
                        "information_extraction", "text_generation", "sentiment_analysis"
                    ],
                    version="1.0"
                ),
                Certificate(
                    id="cv_specialist_certificate",
                    name="Computer Vision Specialist Certificate",
                    type=CertificateType.SKILL_CERTIFICATE,
                    description="Specialized certification in computer vision and image processing",
                    requirements={
                        "section_completion": "04_Computer_Vision",
                        "practical_projects": 3,
                        "deployment_projects": 1,
                        "minimum_score": 80
                    },
                    template_id="skill_template",
                    validity_days=730,
                    skills_certified=[
                        "image_processing", "object_detection", "image_segmentation",
                        "3d_vision", "video_analysis"
                    ],
                    version="1.0"
                ),
                Certificate(
                    id="ai_researcher_certificate",
                    name="AI Research Certificate",
                    type=CertificateType.RESEARCH_CERTIFICATE,
                    description="Certificate for research contributions in artificial intelligence",
                    requirements={
                        "research_projects": 2,
                        "paper_submissions": 1,
                        "peer_reviews": 5,
                        "innovation_score": 80
                    },
                    template_id="research_template",
                    validity_days=None,  # No expiration
                    skills_certified=[
                        "research_methodology", "experimental_design", "data_analysis",
                        "academic_writing", "critical_thinking", "innovation"
                    ],
                    version="1.0"
                ),
                Certificate(
                    id="ai_mastery_certificate",
                    name="AI Mastery Certificate",
                    type=CertificateType.MASTERY_CERTIFICATE,
                    description="Highest level certification demonstrating comprehensive AI expertise",
                    requirements={
                        "all_sections_completed": True,
                        "minimum_average_score": 85,
                        "capstone_project": 1,
                        "teaching_contribution": 1,
                        "industry_application": 1
                    },
                    template_id="mastery_template",
                    validity_days=None,  # No expiration
                    skills_certified=[
                        "comprehensive_ai_knowledge", "practical_implementation",
                        "research_capability", "industry_application", "teaching_mentorship"
                    ],
                    version="1.0"
                )
            ]

            for certificate in certificates:
                self.certificates[certificate.id] = certificate

    def check_achievements(self, user_id: str, user_data: Dict[str, Any]) -> List[Achievement]:
        """Check and award achievements based on user data"""
        newly_earned = []

        # Initialize user achievements if not exists
        if user_id not in self.user_achievements:
            self.user_achievements[user_id] = {}

        existing_achievements = set(self.user_achievements[user_id].keys())

        for achievement_id, achievement in self.achievements.items():
            if achievement_id in existing_achievements:
                continue  # Already earned

            # Check prerequisites
            if achievement.prerequisites:
                if not all(prereq in existing_achievements for prereq in achievement.prerequisites):
                    continue

            # Check requirements
            if self._check_achievement_requirements(achievement, user_data):
                self._award_achievement(user_id, achievement_id)
                newly_earned.append(achievement)

        if newly_earned:
            self.save_data()
            self._clear_leaderboard_cache()

        return newly_earned

    def _check_achievement_requirements(self, achievement: Achievement, user_data: Dict[str, Any]) -> bool:
        """Check if user meets achievement requirements"""
        requirements = achievement.requirements

        # Check assessments completed
        if "assessments_completed" in requirements:
            user_assessments = user_data.get("assessments_completed", 0)
            if user_assessments < requirements["assessments_completed"]:
                return False

        # Check average score
        if "average_score" in requirements:
            user_avg_score = user_data.get("average_score", 0)
            if user_avg_score < requirements["average_score"]:
                return False

        # Check consecutive days
        if "consecutive_days" in requirements:
            consecutive_days = user_data.get("consecutive_days", 0)
            if consecutive_days < requirements["consecutive_days"]:
                return False

        # Check perfect scores
        if "perfect_scores" in requirements:
            perfect_scores = user_data.get("perfect_scores", 0)
            if perfect_scores < requirements["perfect_scores"]:
                return False

        # Check section completion
        if "section_completion" in requirements:
            completed_sections = user_data.get("completed_sections", [])
            if requirements["section_completion"] not in completed_sections:
                return False

        # Check total assessments
        if "total_assessments" in requirements:
            total_assessments = user_data.get("total_assessments", 0)
            if total_assessments < requirements["total_assessments"]:
                return False

        # Check skill mastery
        if "skill_mastery" in requirements:
            mastered_skills = user_data.get("mastered_skills", [])
            if len(mastered_skills) < requirements["skill_mastery"]:
                return False

        # Check timeframe requirements
        if "timeframe_days" in requirements:
            timeframe_days = requirements["timeframe_days"]
            # This would need implementation based on timestamp data
            pass

        return True

    def _award_achievement(self, user_id: str, achievement_id: str):
        """Award an achievement to a user"""
        achievement = self.achievements[achievement_id]

        user_achievement = UserAchievement(
            user_id=user_id,
            achievement_id=achievement_id,
            earned_at=datetime.now(),
            expires_at=(datetime.now() + timedelta(days=achievement.expiration_days)) if achievement.expiration_days else None,
            progress={"completion": 100.0},
            metadata={"awarded_automatically": True}
        )

        self.user_achievements[user_id][achievement_id] = asdict(user_achievement)

    def check_certificates(self, user_id: str, user_data: Dict[str, Any]) -> List[Certificate]:
        """Check and award certificates based on user data"""
        newly_earned = []

        # Initialize user certificates if not exists
        if user_id not in self.user_certificates:
            self.user_certificates[user_id] = {}

        existing_certificates = set(self.user_certificates[user_id].keys())

        for certificate_id, certificate in self.certificates.items():
            if certificate_id in existing_certificates:
                continue  # Already earned

            # Check requirements
            if self._check_certificate_requirements(certificate, user_data):
                self._award_certificate(user_id, certificate_id)
                newly_earned.append(certificate)

        if newly_earned:
            self.save_data()

        return newly_earned

    def _check_certificate_requirements(self, certificate: Certificate, user_data: Dict[str, Any]) -> bool:
        """Check if user meets certificate requirements"""
        requirements = certificate.requirements

        # Check section completion
        if "section_completion" in requirements:
            completed_sections = user_data.get("completed_sections", [])
            if requirements["section_completion"] not in completed_sections:
                return False

        # Check multiple section completions
        if "section_completions" in requirements:
            required_sections = requirements["section_completions"]
            completed_sections = user_data.get("completed_sections", [])
            if not all(section in completed_sections for section in required_sections):
                return False

        # Check minimum score
        if "minimum_score" in requirements:
            user_avg_score = user_data.get("average_score", 0)
            if user_avg_score < requirements["minimum_score"]:
                return False

        # Check required assessments
        if "required_assessments" in requirements:
            user_assessments = user_data.get("assessments_completed", 0)
            if user_assessments < requirements["required_assessments"]:
                return False

        # Check practical projects
        if "practical_projects" in requirements:
            practical_projects = user_data.get("practical_projects", 0)
            if practical_projects < requirements["practical_projects"]:
                return False

        # Check capstone project
        if "capstone_project" in requirements:
            capstone_completed = user_data.get("capstone_completed", False)
            if not capstone_completed:
                return False

        # Check all sections completion
        if "all_sections_completed" in requirements:
            total_sections = 25  # Based on project structure
            completed_sections = user_data.get("completed_sections", [])
            if len(completed_sections) < total_sections:
                return False

        return True

    def _award_certificate(self, user_id: str, certificate_id: str):
        """Award a certificate to a user"""
        certificate = self.certificates[certificate_id]

        # Generate unique verification code
        verification_code = self._generate_verification_code(user_id, certificate_id)

        # Generate credential ID
        credential_id = self._generate_credential_id(user_id, certificate_id)

        user_certificate = UserCertificate(
            user_id=user_id,
            certificate_id=certificate_id,
            issued_at=datetime.now(),
            expires_at=(datetime.now() + timedelta(days=certificate.validity_days)) if certificate.validity_days else None,
            verification_code=verification_code,
            credential_id=credential_id,
            status="active",
            metadata={"awarded_automatically": True}
        )

        self.user_certificates[user_id][certificate_id] = asdict(user_certificate)

    def _generate_verification_code(self, user_id: str, certificate_id: str) -> str:
        """Generate unique verification code for certificate"""
        timestamp = str(int(time.time()))
        data = f"{user_id}_{certificate_id}_{timestamp}"
        hash_object = hashlib.sha256(data.encode())
        return hash_object.hexdigest()[:12].upper()

    def _generate_credential_id(self, user_id: str, certificate_id: str) -> str:
        """Generate unique credential ID"""
        uuid_str = str(uuid.uuid4())
        return f"CRED-{uuid_str[:8].upper()}"

    def get_user_achievements(self, user_id: str) -> Dict[str, Any]:
        """Get all achievements for a user"""
        if user_id not in self.user_achievements:
            return {"user_id": user_id, "achievements": [], "total_points": 0}

        user_achievements_data = self.user_achievements[user_id]
        achievements_list = []
        total_points = 0

        for achievement_id, achievement_data in user_achievements_data.items():
            if achievement_id in self.achievements:
                achievement = self.achievements[achievement_id]
                achievements_list.append({
                    "achievement": achievement,
                    "earned_at": achievement_data.get("earned_at"),
                    "expires_at": achievement_data.get("expires_at"),
                    "progress": achievement_data.get("progress", {})
                })
                total_points += achievement.points_value

        return {
            "user_id": user_id,
            "achievements": achievements_list,
            "total_points": total_points,
            "achievement_count": len(achievements_list)
        }

    def get_user_certificates(self, user_id: str) -> Dict[str, Any]:
        """Get all certificates for a user"""
        if user_id not in self.user_certificates:
            return {"user_id": user_id, "certificates": [], "active_count": 0}

        user_certificates_data = self.user_certificates[user_id]
        certificates_list = []
        active_count = 0

        for certificate_id, certificate_data in user_certificates_data.items():
            if certificate_id in self.certificates:
                certificate = self.certificates[certificate_id]
                certificates_list.append({
                    "certificate": certificate,
                    "issued_at": certificate_data.get("issued_at"),
                    "expires_at": certificate_data.get("expires_at"),
                    "verification_code": certificate_data.get("verification_code"),
                    "credential_id": certificate_data.get("credential_id"),
                    "status": certificate_data.get("status", "active")
                })

                if certificate_data.get("status", "active") == "active":
                    active_count += 1

        return {
            "user_id": user_id,
            "certificates": certificates_list,
            "active_count": active_count,
            "total_count": len(certificates_list)
        }

    def get_certificate_details(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Get certificate details by credential ID for verification"""
        # Search across all users
        for user_id, user_certs in self.user_certificates.items():
            for cert_data in user_certs.values():
                if cert_data.get("credential_id") == credential_id:
                    certificate_id = cert_data.get("certificate_id")
                    if certificate_id in self.certificates:
                        certificate = self.certificates[certificate_id]
                        return {
                            "certificate": certificate,
                            "user_id": user_id,
                            "issued_at": cert_data.get("issued_at"),
                            "expires_at": cert_data.get("expires_at"),
                            "verification_code": cert_data.get("verification_code"),
                            "status": cert_data.get("status", "active")
                        }
        return None

    def verify_certificate(self, verification_code: str) -> Optional[Dict[str, Any]]:
        """Verify certificate by verification code"""
        # Search across all users
        for user_id, user_certs in self.user_certificates.items():
            for cert_data in user_certs.values():
                if cert_data.get("verification_code") == verification_code:
                    certificate_id = cert_data.get("certificate_id")
                    if certificate_id in self.certificates:
                        certificate = self.certificates[certificate_id]
                        return {
                            "valid": True,
                            "certificate": certificate,
                            "user_id": user_id,
                            "issued_at": cert_data.get("issued_at"),
                            "expires_at": cert_data.get("expires_at"),
                            "status": cert_data.get("status", "active")
                        }

        return {"valid": False, "message": "Certificate not found or invalid"}

    def generate_leaderboard(self, category: str = "overall", limit: int = 50) -> List[LeaderboardEntry]:
        """Generate leaderboard for different categories"""
        cache_key = f"{category}_{limit}"
        if cache_key in self.leaderboard_cache:
            cache_time, cache_data = self.leaderboard_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < 3600:  # 1 hour cache
                return cache_data

        leaderboard = []

        # Get all users with achievements/certificates
        all_users = set(self.user_achievements.keys()) | set(self.user_certificates.keys())

        for user_id in all_users:
            user_achievements = self.get_user_achievements(user_id)
            user_certificates = self.get_user_certificates(user_id)

            # Calculate score based on category
            if category == "overall":
                score = (
                    user_achievements["total_points"] * 1.0 +
                    user_certificates["active_count"] * 500 +
                    user_certificates["total_count"] * 200
                )
            elif category == "achievements":
                score = user_achievements["total_points"]
            elif category == "certificates":
                score = user_certificates["active_count"] * 1000 + user_certificates["total_count"] * 500
            else:
                score = user_achievements["total_points"]

            # Get special badges (rare achievements)
            special_badges = []
            for achievement_data in user_achievements["achievements"]:
                achievement = achievement_data["achievement"]
                if achievement.tier in [BadgeTier.PLATINUM, BadgeTier.DIAMOND]:
                    special_badges.append(achievement.id)

            entry = LeaderboardEntry(
                user_id=user_id,
                display_name=user_id,  # In real system, would get from user profile
                score=score,
                rank=0,  # Will be calculated after sorting
                achievements_count=user_achievements["achievement_count"],
                certificates_count=user_certificates["active_count"],
                level=1,  # Would calculate from user data
                special_badges=special_badges
            )

            leaderboard.append(entry)

        # Sort by score and assign ranks
        leaderboard.sort(key=lambda x: x.score, reverse=True)
        for i, entry in enumerate(leaderboard):
            entry.rank = i + 1

        # Cache result
        self.leaderboard_cache[cache_key] = (datetime.now(), leaderboard[:limit])

        return leaderboard[:limit]

    def _clear_leaderboard_cache(self):
        """Clear leaderboard cache when data changes"""
        self.leaderboard_cache.clear()

    def get_achievement_statistics(self) -> Dict[str, Any]:
        """Get global achievement statistics"""
        total_users = len(set(self.user_achievements.keys()) | set(self.user_certificates.keys()))

        # Achievement statistics
        achievement_stats = {}
        for achievement_id, achievement in self.achievements.items():
            earned_count = sum(1 for user_achievements in self.user_achievements.values()
                             if achievement_id in user_achievements)
            achievement_stats[achievement_id] = {
                "name": achievement.name,
                "earned_count": earned_count,
                "earn_rate": earned_count / total_users if total_users > 0 else 0,
                "tier": achievement.tier.value,
                "points_value": achievement.points_value
            }

        # Certificate statistics
        certificate_stats = {}
        for certificate_id, certificate in self.certificates.items():
            earned_count = sum(1 for user_certificates in self.user_certificates.values()
                             if certificate_id in user_certificates)
            certificate_stats[certificate_id] = {
                "name": certificate.name,
                "earned_count": earned_count,
                "earn_rate": earned_count / total_users if total_users > 0 else 0,
                "type": certificate.type.value
            }

        return {
            "total_users": total_users,
            "total_achievements_earned": sum(len(achievements) for achievements in self.user_achievements.values()),
            "total_certificates_earned": sum(len(certificates) for certificates in self.user_certificates.values()),
            "achievement_statistics": achievement_stats,
            "certificate_statistics": certificate_stats
        }

    def generate_progress_report(self, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive progress report for a user"""
        user_achievements = self.get_user_achievements(user_id)
        user_certificates = self.get_user_certificates(user_id)

        # Calculate next achievable achievements
        next_achievements = []
        for achievement_id, achievement in self.achievements.items():
            if achievement_id not in [a["achievement"].id for a in user_achievements["achievements"]]:
                # Check if prerequisites are met
                if all(prereq in [a["achievement"].id for a in user_achievements["achievements"]]
                       for prereq in achievement.prerequisites):
                    next_achievements.append(achievement)

        # Calculate next achievable certificates
        next_certificates = []
        for certificate_id, certificate in self.certificates.items():
            if certificate_id not in [c["certificate"].id for c in user_certificates["certificates"]]:
                next_certificates.append(certificate)

        # Get user rank
        leaderboard = self.generate_leaderboard("overall", limit=1000)
        user_rank = next((entry.rank for entry in leaderboard if entry.user_id == user_id), len(leaderboard) + 1)

        return {
            "user_id": user_id,
            "achievement_summary": {
                "total_points": user_achievements["total_points"],
                "achievement_count": user_achievements["achievement_count"],
                "rarest_achievement": self._get_rarest_achievement(user_achievements["achievements"]),
                "highest_tier": self._get_highest_tier(user_achievements["achievements"])
            },
            "certificate_summary": {
                "active_count": user_certificates["active_count"],
                "total_count": user_certificates["total_count"],
                "highest_level_certificate": self._get_highest_level_certificate(user_certificates["certificates"])
            },
            "next_goals": {
                "next_achievements": next_achievements[:5],  # Top 5 next achievements
                "next_certificates": next_certificates[:3]  # Top 3 next certificates
            },
            "ranking": {
                "global_rank": user_rank,
                "total_users": len(leaderboard),
                "percentile": (len(leaderboard) - user_rank + 1) / len(leaderboard) * 100 if leaderboard else 0
            },
            "recent_activity": self._get_recent_activity(user_id)
        }

    def _get_rarest_achievement(self, achievements: List[Dict]) -> Optional[str]:
        """Get the rarest achievement earned by user"""
        if not achievements:
            return None

        rarest = min(achievements, key=lambda x: x["achievement"].rarity)
        return rarest["achievement"].name

    def _get_highest_tier(self, achievements: List[Dict]) -> Optional[str]:
        """Get the highest tier achievement earned by user"""
        if not achievements:
            return None

        tier_order = {BadgeTier.DIAMOND: 5, BadgeTier.PLATINUM: 4, BadgeTier.GOLD: 3,
                     BadgeTier.SILVER: 2, BadgeTier.BRONZE: 1}

        highest = max(achievements, key=lambda x: tier_order.get(x["achievement"].tier, 0))
        return highest["achievement"].tier.value

    def _get_highest_level_certificate(self, certificates: List[Dict]) -> Optional[str]:
        """Get the highest level certificate earned by user"""
        if not certificates:
            return None

        type_order = {CertificateType.MASTERY_CERTIFICATE: 5, CertificateType.RESEARCH_CERTIFICATE: 4,
                     CertificateType.PROFESSIONAL_CERTIFICATE: 3, CertificateType.SKILL_CERTIFICATE: 2,
                     CertificateType.SECTION_CERTIFICATE: 1}

        highest = max(certificates, key=lambda x: type_order.get(x["certificate"].type, 0))
        return highest["certificate"].name

    def _get_recent_activity(self, user_id: str) -> List[Dict]:
        """Get recent achievement and certificate activity"""
        recent_activity = []

        # Recent achievements
        if user_id in self.user_achievements:
            for achievement_id, achievement_data in self.user_achievements[user_id].items():
                earned_at = datetime.fromisoformat(achievement_data["earned_at"])
                if (datetime.now() - earned_at).days <= 30:  # Last 30 days
                    achievement = self.achievements.get(achievement_id)
                    if achievement:
                        recent_activity.append({
                            "type": "achievement",
                            "name": achievement.name,
                            "earned_at": earned_at,
                            "tier": achievement.tier.value,
                            "points": achievement.points_value
                        })

        # Recent certificates
        if user_id in self.user_certificates:
            for certificate_id, certificate_data in self.user_certificates[user_id].items():
                issued_at = datetime.fromisoformat(certificate_data["issued_at"])
                if (datetime.now() - issued_at).days <= 30:  # Last 30 days
                    certificate = self.certificates.get(certificate_id)
                    if certificate:
                        recent_activity.append({
                            "type": "certificate",
                            "name": certificate.name,
                            "issued_at": issued_at,
                            "type_name": certificate.type.value,
                            "verification_code": certificate_data.get("verification_code", "")
                        })

        # Sort by date
        recent_activity.sort(key=lambda x: x["earned_at"] if "earned_at" in x else x["issued_at"], reverse=True)
        return recent_activity[:10]  # Last 10 activities

    def _achievement_to_dict(self, achievement: Achievement) -> Dict:
        """Convert Achievement to dictionary for JSON serialization"""
        return {
            'id': achievement.id,
            'name': achievement.name,
            'description': achievement.description,
            'type': achievement.type.value,
            'tier': achievement.tier.value,
            'icon_url': achievement.icon_url,
            'requirements': achievement.requirements,
            'points_value': achievement.points_value,
            'rarity': achievement.rarity,
            'category': achievement.category,
            'expiration_days': achievement.expiration_days,
            'prerequisites': achievement.prerequisites
        }

    def _dict_to_achievement(self, data: Dict) -> Achievement:
        """Convert dictionary to Achievement object"""
        return Achievement(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            type=AchievementType(data['type']),
            tier=BadgeTier(data['tier']),
            icon_url=data['icon_url'],
            requirements=data['requirements'],
            points_value=data['points_value'],
            rarity=data['rarity'],
            category=data['category'],
            expiration_days=data.get('expiration_days'),
            prerequisites=data.get('prerequisites', [])
        )

    def _certificate_to_dict(self, certificate: Certificate) -> Dict:
        """Convert Certificate to dictionary for JSON serialization"""
        return {
            'id': certificate.id,
            'name': certificate.name,
            'type': certificate.type.value,
            'description': certificate.description,
            'requirements': certificate.requirements,
            'template_id': certificate.template_id,
            'validity_days': certificate.validity_days,
            'issuer': certificate.issuer,
            'skills_certified': certificate.skills_certified,
            'version': certificate.version
        }

    def _dict_to_certificate(self, data: Dict) -> Certificate:
        """Convert dictionary to Certificate object"""
        return Certificate(
            id=data['id'],
            name=data['name'],
            type=CertificateType(data['type']),
            description=data['description'],
            requirements=data['requirements'],
            template_id=data['template_id'],
            validity_days=data.get('validity_days'),
            issuer=data.get('issuer', 'AI Documentation Project'),
            skills_certified=data.get('skills_certified', []),
            version=data.get('version', '1.0')
        )

if __name__ == "__main__":
    # Example usage
    achievement_system = AchievementSystem()

    # Sample user data
    user_data = {
        "user_id": "user_001",
        "assessments_completed": 15,
        "average_score": 87,
        "consecutive_days": 5,
        "perfect_scores": 2,
        "completed_sections": ["01_Foundational_Machine_Learning"],
        "total_assessments": 15,
        "mastered_skills": ["linear_algebra"],
        "practical_projects": 2,
        "completed_sections": ["01_Foundational_Machine_Learning"]
    }

    # Check for achievements
    new_achievements = achievement_system.check_achievements("user_001", user_data)
    print(f"New achievements earned: {len(new_achievements)}")
    for achievement in new_achievements:
        print(f"  - {achievement.name} ({achievement.tier.value})")

    # Check for certificates
    new_certificates = achievement_system.check_certificates("user_001", user_data)
    print(f"New certificates earned: {len(new_certificates)}")
    for certificate in new_certificates:
        print(f"  - {certificate.name}")

    # Get user progress report
    progress_report = achievement_system.generate_progress_report("user_001")
    print(f"\nProgress Report for user_001:")
    print(f"  Total Points: {progress_report['achievement_summary']['total_points']}")
    print(f"  Achievements: {progress_report['achievement_summary']['achievement_count']}")
    print(f"  Active Certificates: {progress_report['certificate_summary']['active_count']}")
    print(f"  Global Rank: {progress_report['ranking']['global_rank']}")

    # Get global statistics
    stats = achievement_system.get_achievement_statistics()
    print(f"\nGlobal Statistics:")
    print(f"  Total Users: {stats['total_users']}")
    print(f"  Total Achievements Earned: {stats['total_achievements_earned']}")
    print(f"  Total Certificates Earned: {stats['total_certificates_earned']}")

    # Generate leaderboard
    leaderboard = achievement_system.generate_leaderboard("overall", limit=10)
    print(f"\nTop 10 Leaderboard:")
    for i, entry in enumerate(leaderboard[:5]):
        print(f"  {i+1}. {entry.user_id} - Score: {entry.score:.0f}")

    print("\nAchievement and certification system initialized successfully!")