#!/usr/bin/env python3
"""
Advanced Gamification Engine for Interactive Learning
Comprehensive achievement system with streaks, badges, leaderboards, and rewards
"""

import json
import os
import sys
import uuid
import random
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)

class BadgeTier(Enum):
    """Badge tier levels"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"
    LEGENDARY = "legendary"

class AchievementType(Enum):
    """Types of achievements"""
    SKILL_MASTERY = "skill_mastery"
    STREAK_ACHIEVEMENT = "streak_achievement"
    SPEED_ACHIEVEMENT = "speed_achievement"
    COLLABORATION_ACHIEVEMENT = "collaboration_achievement"
    EXPLORATION_ACHIEVEMENT = "exploration_achievement"
    PERFECT_SCORE_ACHIEVEMENT = "perfect_score_achievement"
    CONSISTENCY_ACHIEVEMENT = "consistency_achievement"
    COMMUNITY_ACHIEVEMENT = "community_achievement"
    INNOVATION_ACHIEVEMENT = "innovation_achievement"
    MILESTONE_ACHIEVEMENT = "milestone_achievement"

class RewardType(Enum):
    """Types of rewards"""
    EXPERIENCE_POINTS = "experience_points"
    VIRTUAL_CURRENCY = "virtual_currency"
    CUSTOMIZATION_ITEM = "customization_item"
    PREMIUM_ACCESS = "premium_access"
    MENTORSHIP_SESSION = "mentorship_session"
    CERTIFICATE = "certificate"
    BADGE = "badge"
    LEADERBOARD_BOOST = "leaderboard_boost"

@dataclass
class Badge:
    """Badge definition"""
    badge_id: str
    name: str
    description: str
    tier: BadgeTier
    category: str
    icon_url: str
    rarity: float  # 0.0 - 1.0, lower is rarer
    points_value: int
    unlock_conditions: Dict[str, Any]
    hidden: bool = False
    limited_edition: bool = False
    expiry_date: Optional[datetime] = None

@dataclass
class Achievement:
    """Achievement definition"""
    achievement_id: str
    name: str
    description: str
    achievement_type: AchievementType
    badge_id: str
    points_reward: int
    experience_reward: int
    unlock_conditions: Dict[str, Any]
    progress_tracking: bool
    multiple_awards: bool = False
    secret: bool = False

@dataclass
class UserAchievement:
    """User's earned achievement"""
    user_id: str
    achievement_id: str
    earned_at: datetime
    progress_data: Dict[str, Any]
    badge_earned: Badge
    points_earned: int
    experience_earned: int
    share_count: int = 0
    display_priority: int = 0

@dataclass
class LearningStreak:
    """Learning streak tracking"""
    user_id: str
    current_streak_days: int
    longest_streak_days: int
    last_activity_date: datetime
    streak_start_date: datetime
    milestone_rewards_claimed: List[int]
    streak_multiplier: float = 1.0
    fire_intensity: str = "warm"  # cold, warm, hot, burning, inferno

@dataclass
class UserProfile:
    """Extended user profile for gamification"""
    user_id: str
    display_name: str
    avatar_url: str
    level: int
    experience_points: int
    virtual_currency: int
    total_badges: int
    favorite_badges: List[str]
    showcase_items: List[str]
    title: str
    rank: str
    reputation_score: int
    created_at: datetime
    last_active: datetime

@dataclass
class LeaderboardEntry:
    """Leaderboard entry"""
    user_id: str
    display_name: str
    avatar_url: str
    score: int
    rank: int
    change: int  # Rank change from previous period
    badges_count: int
    level: int
    special_achievements: List[str]

class GamificationEngine:
    """Advanced gamification engine"""

    def __init__(self, data_path: str = "gamification_data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)

        # Initialize data structures
        self.achievements = {}
        self.badges = {}
        self.user_achievements = defaultdict(list)
        self.user_profiles = {}
        self.learning_streaks = {}
        self.leaderboards = {}
        self.reward_history = defaultdict(list)

        # Experience and leveling system
        self.level_thresholds = self._generate_level_thresholds()
        self.experience_multipliers = {
            'perfect_score': 2.0,
            'speed_bonus': 1.5,
            'streak_bonus': 1.3,
            'difficulty_bonus': 1.2,
            'collaboration_bonus': 1.4
        }

        # Load existing data
        self._load_gamification_data()
        self._initialize_default_achievements()
        self._initialize_default_badges()

        logger.info("Gamification Engine initialized")

    def _generate_level_thresholds(self) -> Dict[int, int]:
        """Generate experience thresholds for each level"""
        thresholds = {}
        base_xp = 100
        growth_factor = 1.5

        for level in range(1, 101):  # Levels 1-100
            if level == 1:
                thresholds[level] = 0
            else:
                thresholds[level] = int(base_xp * (growth_factor ** (level - 2)))

        return thresholds

    def _initialize_default_achievements(self):
        """Initialize default achievement definitions"""
        default_achievements = [
            # Skill Mastery Achievements
            Achievement(
                achievement_id="first_steps",
                name="First Steps",
                description="Complete your first exercise",
                achievement_type=AchievementType.MILESTONE_ACHIEVEMENT,
                badge_id="beginner_badge",
                points_reward=10,
                experience_reward=50,
                unlock_conditions={"exercises_completed": 1},
                progress_tracking=True
            ),
            Achievement(
                achievement_id="skill_dedication",
                name="Skill Dedication",
                description="Complete 50 exercises in a single section",
                achievement_type=AchievementType.SKILL_MASTERY,
                badge_id="dedication_badge",
                points_reward=100,
                experience_reward=500,
                unlock_conditions={"section_exercises": 50},
                progress_tracking=True
            ),
            Achievement(
                achievement_id="section_master",
                name="Section Master",
                description="Achieve 90% mastery in any section",
                achievement_type=AchievementType.SKILL_MASTERY,
                badge_id="mastery_badge",
                points_reward=250,
                experience_reward=1000,
                unlock_conditions={"section_mastery": 0.9},
                progress_tracking=True
            ),

            # Streak Achievements
            Achievement(
                achievement_id="week_warrior",
                name="Week Warrior",
                description="Maintain a 7-day learning streak",
                achievement_type=AchievementType.STREAK_ACHIEVEMENT,
                badge_id="streak_7_badge",
                points_reward=50,
                experience_reward=200,
                unlock_conditions={"streak_days": 7},
                progress_tracking=True
            ),
            Achievement(
                achievement_id="month_marathon",
                name="Month Marathon",
                description="Maintain a 30-day learning streak",
                achievement_type=AchievementType.STREAK_ACHIEVEMENT,
                badge_id="streak_30_badge",
                points_reward=200,
                experience_reward=1000,
                unlock_conditions={"streak_days": 30},
                progress_tracking=True
            ),

            # Speed Achievements
            Achievement(
                achievement_id="speed_demon",
                name="Speed Demon",
                description="Complete 10 exercises with perfect scores under time limit",
                achievement_type=AchievementType.SPEED_ACHIEVEMENT,
                badge_id="speed_badge",
                points_reward=150,
                experience_reward=750,
                unlock_conditions={"perfect_speed_exercises": 10},
                progress_tracking=True
            ),

            # Collaboration Achievements
            Achievement(
                achievement_id="team_player",
                name="Team Player",
                description="Help 5 other learners with their exercises",
                achievement_type=AchievementType.COLLABORATION_ACHIEVEMENT,
                badge_id="collaboration_badge",
                points_reward=100,
                experience_reward=500,
                unlock_conditions={"peers_helped": 5},
                progress_tracking=True
            ),

            # Perfect Score Achievements
            Achievement(
                achievement_id="perfectionist",
                name="Perfectionist",
                description="Achieve perfect scores on 25 exercises",
                achievement_type=AchievementType.PERFECT_SCORE_ACHIEVEMENT,
                badge_id="perfect_badge",
                points_reward=200,
                experience_reward=1000,
                unlock_conditions={"perfect_scores": 25},
                progress_tracking=True
            )
        ]

        for achievement in default_achievements:
            self.achievements[achievement.achievement_id] = achievement

    def _initialize_default_badges(self):
        """Initialize default badge definitions"""
        default_badges = [
            Badge(
                badge_id="beginner_badge",
                name="Novice Learner",
                description="Started your learning journey",
                tier=BadgeTier.BRONZE,
                category="milestone",
                icon_url="/badges/beginner.svg",
                rarity=0.8,
                points_value=10,
                unlock_conditions={"first_exercise": True}
            ),
            Badge(
                badge_id="dedication_badge",
                name="Dedicated Learner",
                description="Committed to skill development",
                tier=BadgeTier.SILVER,
                category="skill",
                icon_url="/badges/dedication.svg",
                rarity=0.6,
                points_value=100,
                unlock_conditions={"section_exercises": 50}
            ),
            Badge(
                badge_id="mastery_badge",
                name="Section Master",
                description="Achieved mastery in a section",
                tier=BadgeTier.GOLD,
                category="skill",
                icon_url="/badges/mastery.svg",
                rarity=0.4,
                points_value=250,
                unlock_conditions={"section_mastery": 0.9}
            ),
            Badge(
                badge_id="streak_7_badge",
                name="Week Warrior",
                description="7-day learning streak",
                tier=BadgeTier.SILVER,
                category="streak",
                icon_url="/badges/streak_7.svg",
                rarity=0.5,
                points_value=50,
                unlock_conditions={"streak_days": 7}
            ),
            Badge(
                badge_id="streak_30_badge",
                name="Month Marathon",
                description="30-day learning streak",
                tier=BadgeTier.GOLD,
                category="streak",
                icon_url="/badges/streak_30.svg",
                rarity=0.3,
                points_value=200,
                unlock_conditions={"streak_days": 30}
            ),
            Badge(
                badge_id="speed_badge",
                name="Speed Demon",
                description="Master of speed learning",
                tier=BadgeTier.PLATINUM,
                category="speed",
                icon_url="/badges/speed.svg",
                rarity=0.2,
                points_value=150,
                unlock_conditions={"perfect_speed_exercises": 10}
            ),
            Badge(
                badge_id="collaboration_badge",
                name="Team Player",
                description="Excellent collaborator",
                tier=BadgeTier.SILVER,
                category="collaboration",
                icon_url="/badges/collaboration.svg",
                rarity=0.5,
                points_value=100,
                unlock_conditions={"peers_helped": 5}
            ),
            Badge(
                badge_id="perfect_badge",
                name="Perfectionist",
                description="Master of perfection",
                tier=BadgeTier.PLATINUM,
                category="performance",
                icon_url="/badges/perfect.svg",
                rarity=0.15,
                points_value=200,
                unlock_conditions={"perfect_scores": 25}
            )
        ]

        for badge in default_badges:
            self.badges[badge.badge_id] = badge

    def create_user_profile(self, user_id: str, display_name: str, avatar_url: str = None) -> UserProfile:
        """Create user profile for gamification"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        profile = UserProfile(
            user_id=user_id,
            display_name=display_name,
            avatar_url=avatar_url or f"/avatars/default/{user_id}.png",
            level=1,
            experience_points=0,
            virtual_currency=100,  # Starting currency
            total_badges=0,
            favorite_badges=[],
            showcase_items=[],
            title="Novice Learner",
            rank="Bronze",
            reputation_score=0,
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc)
        )

        self.user_profiles[user_id] = profile
        self.learning_streaks[user_id] = LearningStreak(
            user_id=user_id,
            current_streak_days=0,
            longest_streak_days=0,
            last_activity_date=datetime.now(timezone.utc),
            streak_start_date=datetime.now(timezone.utc),
            milestone_rewards_claimed=[]
        )

        self._save_user_profiles()
        logger.info(f"Created gamification profile for user: {user_id}")
        return profile

    def award_experience_points(self, user_id: str, base_points: int, modifiers: Dict[str, float] = None) -> int:
        """Award experience points with modifiers"""
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id, user_id)

        profile = self.user_profiles[user_id]

        # Apply modifiers
        final_points = base_points
        if modifiers:
            for modifier_type, multiplier in modifiers.items():
                if modifier_type in self.experience_multipliers:
                    final_points = int(final_points * self.experience_multipliers[modifier_type])

        # Add streak bonus
        streak_bonus = self._calculate_streak_bonus(user_id)
        final_points = int(final_points * streak_bonus)

        # Update profile
        profile.experience_points += final_points

        # Check for level up
        old_level = profile.level
        new_level = self._calculate_level(profile.experience_points)
        profile.level = new_level

        # Award level up rewards
        if new_level > old_level:
            self._award_level_up_rewards(user_id, old_level, new_level)

        profile.last_active = datetime.now(timezone.utc)
        self._save_user_profiles()

        # Record in reward history
        self.reward_history[user_id].append({
            'type': 'experience_points',
            'amount': final_points,
            'base_amount': base_points,
            'modifiers': modifiers or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        logger.info(f"Awarded {final_points} experience points to user: {user_id}")
        return final_points

    def check_and_award_achievements(self, user_id: str, user_data: Dict[str, Any]) -> List[UserAchievement]:
        """Check for and award new achievements based on user data"""
        new_achievements = []

        if user_id not in self.user_profiles:
            self.create_user_profile(user_id, user_id)

        # Update streak first
        self._update_learning_streak(user_id)

        for achievement_id, achievement in self.achievements.items():
            # Skip if already earned and not multiple awards
            if not achievement.multiple_awards:
                already_earned = any(
                    ua.achievement_id == achievement_id
                    for ua in self.user_achievements[user_id]
                )
                if already_earned:
                    continue

            # Check unlock conditions
            if self._check_achievement_conditions(achievement.unlock_conditions, user_data):
                # Award achievement
                badge = self.badges.get(achievement.badge_id)
                if badge:
                    user_achievement = UserAchievement(
                        user_id=user_id,
                        achievement_id=achievement_id,
                        earned_at=datetime.now(timezone.utc),
                        progress_data=user_data,
                        badge_earned=badge,
                        points_earned=achievement.points_reward,
                        experience_earned=achievement.experience_reward,
                        display_priority=self._calculate_display_priority(achievement)
                    )

                    self.user_achievements[user_id].append(user_achievement)
                    new_achievements.append(user_achievement)

                    # Award rewards
                    self.award_experience_points(user_id, achievement.experience_reward)
                    self.user_profiles[user_id].virtual_currency += achievement.points_reward
                    self.user_profiles[user_id].total_badges += 1

                    logger.info(f"Awarded achievement '{achievement.name}' to user: {user_id}")

        self._save_user_achievements()
        self._save_user_profiles()

        return new_achievements

    def update_learning_streak(self, user_id: str):
        """Update learning streak for user"""
        self._update_learning_streak(user_id)
        self._save_learning_streaks()

    def get_leaderboard(self, category: str = "experience", time_period: str = "all_time", limit: int = 50) -> List[LeaderboardEntry]:
        """Get leaderboard data"""
        if category == "experience":
            return self._get_experience_leaderboard(time_period, limit)
        elif category == "streak":
            return self._get_streak_leaderboard(limit)
        elif category == "badges":
            return self._get_badges_leaderboard(limit)
        elif category == "level":
            return self._get_level_leaderboard(limit)
        else:
            return self._get_experience_leaderboard(time_period, limit)

    def get_user_achievements_display(self, user_id: str) -> Dict[str, Any]:
        """Get formatted achievements display for user"""
        if user_id not in self.user_achievements:
            return {
                'total_achievements': 0,
                'badges': [],
                'recent_achievements': [],
                'favorite_achievements': [],
                'progress_tracking': {}
            }

        user_achievements = self.user_achievements[user_id]
        user_profile = self.user_profiles.get(user_id)

        # Sort by display priority and earned date
        sorted_achievements = sorted(
            user_achievements,
            key=lambda x: (x.display_priority, x.earned_at),
            reverse=True
        )

        # Group badges by tier
        badges_by_tier = defaultdict(list)
        for ua in sorted_achievements:
            badges_by_tier[ua.badge_earned.tier.value].append(ua.badge_earned)

        # Recent achievements (last 10)
        recent_achievements = sorted_achievements[:10]

        # Progress tracking for incomplete achievements
        progress_tracking = self._get_achievement_progress(user_id)

        return {
            'total_achievements': len(user_achievements),
            'total_points': sum(ua.points_earned for ua in user_achievements),
            'total_experience': sum(ua.experience_earned for ua in user_achievements),
            'badges_by_tier': dict(badges_by_tier),
            'recent_achievements': [self._format_achievement_display(ua) for ua in recent_achievements],
            'favorite_achievements': [],  # Would be based on user preferences
            'progress_tracking': progress_tracking,
            'achievement_categories': self._get_achievement_categories(user_achievements),
            'rarest_badges': self._get_rarest_badges(user_achievements)
        }

    def _update_learning_streak(self, user_id: str):
        """Update learning streak for user"""
        if user_id not in self.learning_streaks:
            self.learning_streaks[user_id] = LearningStreak(
                user_id=user_id,
                current_streak_days=0,
                longest_streak_days=0,
                last_activity_date=datetime.now(timezone.utc),
                streak_start_date=datetime.now(timezone.utc),
                milestone_rewards_claimed=[]
            )

        streak = self.learning_streaks[user_id]
        now = datetime.now(timezone.utc)

        # Check if activity was today
        if streak.last_activity_date.date() == now.date():
            return  # Already updated today

        # Check if activity was yesterday
        yesterday = now - timedelta(days=1)
        if streak.last_activity_date.date() == yesterday.date():
            streak.current_streak_days += 1
        else:
            # Streak broken
            if streak.current_streak_days > streak.longest_streak_days:
                streak.longest_streak_days = streak.current_streak_days
            streak.current_streak_days = 1
            streak.streak_start_date = now

        streak.last_activity_date = now
        streak.streak_multiplier = self._calculate_streak_multiplier(streak.current_streak_days)
        streak.fire_intensity = self._get_fire_intensity(streak.current_streak_days)

        # Check for streak milestone rewards
        self._check_streak_milestones(user_id, streak)

    def _calculate_streak_multiplier(self, streak_days: int) -> float:
        """Calculate experience multiplier based on streak"""
        if streak_days >= 100:
            return 2.0
        elif streak_days >= 50:
            return 1.75
        elif streak_days >= 30:
            return 1.5
        elif streak_days >= 14:
            return 1.3
        elif streak_days >= 7:
            return 1.2
        elif streak_days >= 3:
            return 1.1
        else:
            return 1.0

    def _get_fire_intensity(self, streak_days: int) -> str:
        """Get fire intensity based on streak"""
        if streak_days >= 100:
            return "inferno"
        elif streak_days >= 50:
            return "burning"
        elif streak_days >= 21:
            return "hot"
        elif streak_days >= 7:
            return "warm"
        else:
            return "cold"

    def _calculate_level(self, experience_points: int) -> int:
        """Calculate level based on experience points"""
        for level, threshold in sorted(self.level_thresholds.items(), reverse=True):
            if experience_points >= threshold:
                return level
        return 1

    def _check_achievement_conditions(self, conditions: Dict[str, Any], user_data: Dict[str, Any]) -> bool:
        """Check if achievement conditions are met"""
        for condition_key, condition_value in conditions.items():
            if condition_key not in user_data:
                return False

            user_value = user_data[condition_key]

            if isinstance(condition_value, (int, float)):
                if user_value < condition_value:
                    return False
            elif isinstance(condition_value, str):
                if user_value != condition_value:
                    return False
            elif isinstance(condition_value, dict):
                # Handle complex conditions
                if 'operator' in condition_value:
                    operator = condition_value['operator']
                    target = condition_value['value']

                    if operator == 'equals':
                        if user_value != target:
                            return False
                    elif operator == 'greater_than':
                        if user_value <= target:
                            return False
                    elif operator == 'less_than':
                        if user_value >= target:
                            return False

        return True

    def _award_level_up_rewards(self, user_id: str, old_level: int, new_level: int):
        """Award rewards for leveling up"""
        profile = self.user_profiles[user_id]

        # Update title based on level
        if new_level >= 50:
            profile.title = "Master Learner"
            profile.rank = "Diamond"
        elif new_level >= 30:
            profile.title = "Expert Scholar"
            profile.rank = "Platinum"
        elif new_level >= 20:
            profile.title = "Advanced Student"
            profile.rank = "Gold"
        elif new_level >= 10:
            profile.title = "Dedicated Learner"
            profile.rank = "Silver"
        elif new_level >= 5:
            profile.title = "Rising Star"
            profile.rank = "Bronze"

        # Award milestone rewards
        milestone_levels = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
        for milestone in milestone_levels:
            if old_level < milestone <= new_level:
                virtual_currency_reward = milestone * 50
                profile.virtual_currency += virtual_currency_reward

                # Log milestone reward
                self.reward_history[user_id].append({
                    'type': 'level_milestone',
                    'level': milestone,
                    'virtual_currency': virtual_currency_reward,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

    def _get_experience_leaderboard(self, time_period: str, limit: int) -> List[LeaderboardEntry]:
        """Get experience-based leaderboard"""
        entries = []

        for user_id, profile in self.user_profiles.items():
            score = profile.experience_points

            # Filter by time period if needed
            if time_period != "all_time":
                # Would need to implement time-based filtering
                pass

            entries.append(LeaderboardEntry(
                user_id=user_id,
                display_name=profile.display_name,
                avatar_url=profile.avatar_url,
                score=score,
                rank=0,  # Will be set after sorting
                change=0,  # Would calculate from previous period
                badges_count=profile.total_badges,
                level=profile.level,
                special_achievements=[]
            ))

        # Sort and assign ranks
        entries.sort(key=lambda x: x.score, reverse=True)
        for i, entry in enumerate(entries[:limit]):
            entry.rank = i + 1

        return entries[:limit]

    def _calculate_display_priority(self, achievement: Achievement) -> int:
        """Calculate display priority for achievement"""
        base_priority = {
            AchievementType.SKILL_MASTERY: 100,
            AchievementType.STREAK_ACHIEVEMENT: 90,
            AchievementType.PERFECT_SCORE_ACHIEVEMENT: 85,
            AchievementType.SPEED_ACHIEVEMENT: 80,
            AchievementType.COLLABORATION_ACHIEVEMENT: 75,
            AchievementType.MILESTONE_ACHIEVEMENT: 70,
            AchievementType.EXPLORATION_ACHIEVEMENT: 60,
            AchievementType.CONSISTENCY_ACHIEVEMENT: 55,
            AchievementType.COMMUNITY_ACHIEVEMENT: 50,
            AchievementType.INNOVATION_ACHIEVEMENT: 120
        }

        return base_priority.get(achievement.achievement_type, 50)

    def _format_achievement_display(self, user_achievement: UserAchievement) -> Dict[str, Any]:
        """Format achievement for display"""
        return {
            'achievement_id': user_achievement.achievement_id,
            'name': user_achievement.badge_earned.name,
            'description': user_achievement.badge_earned.description,
            'badge': {
                'badge_id': user_achievement.badge_earned.badge_id,
                'tier': user_achievement.badge_earned.tier.value,
                'icon_url': user_achievement.badge_earned.icon_url,
                'rarity': user_achievement.badge_earned.rarity
            },
            'earned_at': user_achievement.earned_at.isoformat(),
            'points_earned': user_achievement.points_earned,
            'experience_earned': user_achievement.experience_earned,
            'share_count': user_achievement.share_count
        }

    def _load_gamification_data(self):
        """Load gamification data from files"""
        # Load user profiles
        profiles_file = self.data_path / "user_profiles.json"
        if profiles_file.exists():
            with open(profiles_file, 'r') as f:
                profiles_data = json.load(f)
                for user_id, profile_data in profiles_data.items():
                    profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                    profile_data['last_active'] = datetime.fromisoformat(profile_data['last_active'])
                    self.user_profiles[user_id] = UserProfile(**profile_data)

        # Load user achievements
        achievements_file = self.data_path / "user_achievements.json"
        if achievements_file.exists():
            with open(achievements_file, 'r') as f:
                achievements_data = json.load(f)
                for user_id, achievements_list in achievements_data.items():
                    for achievement_data in achievements_list:
                        achievement_data['earned_at'] = datetime.fromisoformat(achievement_data['earned_at'])
                        # Reconstruct badge object
                        badge_data = achievement_data.pop('badge_earned')
                        badge = Badge(**badge_data)
                        achievement_data['badge_earned'] = badge
                        self.user_achievements[user_id].append(UserAchievement(**achievement_data))

        # Load learning streaks
        streaks_file = self.data_path / "learning_streaks.json"
        if streaks_file.exists():
            with open(streaks_file, 'r') as f:
                streaks_data = json.load(f)
                for user_id, streak_data in streaks_data.items():
                    streak_data['last_activity_date'] = datetime.fromisoformat(streak_data['last_activity_date'])
                    streak_data['streak_start_date'] = datetime.fromisoformat(streak_data['streak_start_date'])
                    self.learning_streaks[user_id] = LearningStreak(**streak_data)

    def _save_user_profiles(self):
        """Save user profiles to file"""
        profiles_file = self.data_path / "user_profiles.json"
        profiles_data = {}

        for user_id, profile in self.user_profiles.items():
            profile_dict = asdict(profile)
            profile_dict['created_at'] = profile.created_at.isoformat()
            profile_dict['last_active'] = profile.last_active.isoformat()
            profiles_data[user_id] = profile_dict

        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=2)

    def _save_user_achievements(self):
        """Save user achievements to file"""
        achievements_file = self.data_path / "user_achievements.json"
        achievements_data = {}

        for user_id, achievements in self.user_achievements.items():
            achievements_list = []
            for ua in achievements:
                achievement_dict = asdict(ua)
                achievement_dict['earned_at'] = ua.earned_at.isoformat()
                achievement_dict['badge_earned'] = asdict(ua.badge_earned)
                achievements_list.append(achievement_dict)
            achievements_data[user_id] = achievements_list

        with open(achievements_file, 'w') as f:
            json.dump(achievements_data, f, indent=2)

    def _save_learning_streaks(self):
        """Save learning streaks to file"""
        streaks_file = self.data_path / "learning_streaks.json"
        streaks_data = {}

        for user_id, streak in self.learning_streaks.items():
            streak_dict = asdict(streak)
            streak_dict['last_activity_date'] = streak.last_activity_date.isoformat()
            streak_dict['streak_start_date'] = streak.streak_start_date.isoformat()
            streaks_data[user_id] = streak_dict

        with open(streaks_file, 'w') as f:
            json.dump(streaks_data, f, indent=2)

# Additional helper methods would be implemented here...
# (Due to space constraints, showing main structure)

if __name__ == "__main__":
    # Initialize gamification engine
    gamification = GamificationEngine()

    print("Gamification Engine initialized")
    print(f"Achievements loaded: {len(gamification.achievements)}")
    print(f"Badges loaded: {len(gamification.badges)}")
    print(f"User profiles: {len(gamification.user_profiles)}")