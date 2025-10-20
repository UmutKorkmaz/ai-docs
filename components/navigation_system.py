#!/usr/bin/env python3
"""
AI Documentation Navigation System
===================================

Comprehensive navigation and organization system for the AI documentation reader application.
Features smart navigation, content organization, cross-referencing, and progress tracking.

Author: AI Documentation Team
Version: 1.0.0
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class SectionInfo:
    """Information about a documentation section."""
    id: str
    title: str
    description: str
    path: str
    level: int  # 1=Beginner, 2=Intermediate, 3=Advanced, 4=Research/Expert
    category: str
    prerequisites: List[str]
    related_sections: List[str]
    estimated_time: int  # minutes
    topics_count: int
    interactive_notebooks: int
    last_updated: str
    tags: List[str]


@dataclass
class LearningPath:
    """Learning path definition."""
    id: str
    name: str
    description: str
    target_audience: str
    sections: List[str]
    estimated_duration: int  # hours
    difficulty_level: int
    outcomes: List[str]


@dataclass
class NavigationState:
    """User's current navigation state."""
    current_section: Optional[str]
    viewed_sections: Set[str]
    completed_sections: Set[str]
    favorite_sections: Set[str]
    recent_sections: List[str]
    current_path: Optional[str]
    progress_percent: float


class AIDocumentationNavigator:
    """Main navigation system for AI documentation."""

    def __init__(self, base_path: str = "/Users/dtumkorkmaz/Projects/ai-docs"):
        self.base_path = Path(base_path)
        self.sections: Dict[str, SectionInfo] = {}
        self.learning_paths: Dict[str, LearningPath] = {}
        self.cross_references: Dict[str, Set[str]] = {}
        self.knowledge_graph: Dict[str, List[str]] = {}

        # Initialize the navigation system
        self._initialize_sections()
        self._initialize_learning_paths()
        self._build_cross_references()
        self._build_knowledge_graph()

    def _initialize_sections(self):
        """Initialize all 25 AI documentation sections."""
        sections_data = [
            # Section I: Foundational Machine Learning
            SectionInfo(
                id="01_foundational_ml",
                title="Foundational Machine Learning",
                description="Mathematical foundations, core ML concepts, and statistical learning methods",
                path="01_Foundational_Machine_Learning/00_Overview.md",
                level=1,
                category="foundations",
                prerequisites=[],
                related_sections=["02_advanced_dl", "03_nlp", "04_computer_vision"],
                estimated_time=480,
                topics_count=45,
                interactive_notebooks=8,
                last_updated="2024-09-01",
                tags=["mathematics", "statistics", "supervised", "unsupervised", "reinforcement"]
            ),

            # Section II: Advanced Deep Learning
            SectionInfo(
                id="02_advanced_dl",
                title="Advanced Deep Learning",
                description="Neural architectures, specialized systems, and emerging models",
                path="02_Advanced_Deep_Learning/00_Overview.md",
                level=2,
                category="core",
                prerequisites=["01_foundational_ml"],
                related_sections=["01_foundational_ml", "03_nlp", "04_computer_vision", "05_generative_ai"],
                estimated_time=600,
                topics_count=55,
                interactive_notebooks=10,
                last_updated="2024-09-15",
                tags=["neural_networks", "transformers", "attention", "architectures", "optimization"]
            ),

            # Section III: Natural Language Processing
            SectionInfo(
                id="03_nlp",
                title="Natural Language Processing",
                description="Text processing, language models, and advanced NLP applications",
                path="03_Natural_Language_Processing/00_Overview.md",
                level=2,
                category="core",
                prerequisites=["01_foundational_ml", "02_advanced_dl"],
                related_sections=["02_advanced_dl", "05_generative_ai", "06_ai_agents", "13_advanced_security"],
                estimated_time=540,
                topics_count=50,
                interactive_notebooks=9,
                last_updated="2024-09-20",
                tags=["nlp", "language_models", "text_processing", "llms", "prompt_engineering"]
            ),

            # Section IV: Computer Vision
            SectionInfo(
                id="04_computer_vision",
                title="Computer Vision",
                description="Image processing, object detection, and advanced visual AI",
                path="04_Computer_Vision/00_Overview.md",
                level=2,
                category="core",
                prerequisites=["01_foundational_ml", "02_advanced_dl"],
                related_sections=["02_advanced_dl", "05_generative_ai", "15_specialized_apps"],
                estimated_time=520,
                topics_count=48,
                interactive_notebooks=8,
                last_updated="2024-09-18",
                tags=["image_processing", "object_detection", "segmentation", "3d_vision", "medical_imaging"]
            ),

            # Section V: Generative AI
            SectionInfo(
                id="05_generative_ai",
                title="Generative AI",
                description="Foundation models, generative systems, and creative AI applications",
                path="05_Generative_AI/00_Overview.md",
                level=3,
                category="advanced",
                prerequisites=["02_advanced_dl", "03_nlp", "04_computer_vision"],
                related_sections=["03_nlp", "04_computer_vision", "06_ai_agents", "20_entertainment_media"],
                estimated_time=580,
                topics_count=52,
                interactive_notebooks=10,
                last_updated="2024-10-01",
                tags=["generative_models", "llms", "diffusion_models", "creative_ai", "multimodal"]
            ),

            # Section VI: AI Agents and Autonomous Systems
            SectionInfo(
                id="06_ai_agents",
                title="AI Agents and Autonomous Systems",
                description="Autonomous agents, reinforcement learning, and self-directing systems",
                path="06_AI_Agents_and_Autonomous/00_Overview.md",
                level=3,
                category="advanced",
                prerequisites=["01_foundational_ml", "05_generative_ai"],
                related_sections=["05_generative_ai", "15_specialized_apps", "23_aerospace_defense"],
                estimated_time=560,
                topics_count=49,
                interactive_notebooks=9,
                last_updated="2024-09-25",
                tags=["agents", "reinforcement_learning", "autonomous", "robotics", "multi_agent"]
            ),

            # Section VII: AI Ethics and Safety
            SectionInfo(
                id="07_ai_ethics_safety",
                title="AI Ethics and Safety",
                description="Ethical principles, safety research, and responsible AI development",
                path="07_AI_Ethics_and_Safety/00_Overview.md",
                level=2,
                category="foundations",
                prerequisites=[],
                related_sections=["all"],  # Applies to all sections
                estimated_time=400,
                topics_count=35,
                interactive_notebooks=5,
                last_updated="2024-10-02",
                tags=["ethics", "safety", "bias", "fairness", "governance", "alignment"]
            ),

            # Section VIII: AI Applications Industry
            SectionInfo(
                id="08_ai_applications_industry",
                title="AI Applications in Industry",
                description="Real-world AI applications across various industries",
                path="08_AI_Applications_Industry/00_Overview.md",
                level=2,
                category="applications",
                prerequisites=["01_foundational_ml"],
                related_sections=["all"],  # Broad applications
                estimated_time=460,
                topics_count=42,
                interactive_notebooks=7,
                last_updated="2024-09-28",
                tags=["industry", "applications", "healthcare", "finance", "business", "scientific"]
            ),

            # Section IX: Emerging Interdisciplinary
            SectionInfo(
                id="09_emerging_interdisciplinary",
                title="Emerging Interdisciplinary Fields",
                description="AI combined with science, humanities, and engineering",
                path="09_Emerging_Interdisciplinary/00_Overview.md",
                level=3,
                category="research",
                prerequisites=["01_foundational_ml", "02_advanced_dl"],
                related_sections=["all"],
                estimated_time=500,
                topics_count=46,
                interactive_notebooks=8,
                last_updated="2024-09-30",
                tags=["interdisciplinary", "science", "humanities", "engineering", "quantum"]
            ),

            # Section X: Technical Methodological
            SectionInfo(
                id="10_technical_methodological",
                title="Technical and Methodological Advances",
                description="AI systems, infrastructure, and advanced techniques",
                path="10_Technical_Methodological/00_Overview.md",
                level=3,
                category="technical",
                prerequisites=["01_foundational_ml", "02_advanced_dl"],
                related_sections=["all"],
                estimated_time=540,
                topics_count=51,
                interactive_notebooks=9,
                last_updated="2024-10-01",
                tags=["systems", "infrastructure", "mlops", "optimization", "performance"]
            ),

            # Section XI: Future Directions
            SectionInfo(
                id="11_future_directions",
                title="Future Directions and Speculative AI",
                description="AGI, consciousness, and long-term AI research",
                path="11_Future_Directions/00_Overview.md",
                level=4,
                category="research",
                prerequisites=["all"],
                related_sections=["07_ai_ethics_safety"],
                estimated_time=380,
                topics_count=32,
                interactive_notebooks=4,
                last_updated="2024-09-20",
                tags=["agi", "consciousness", "future", "speculative", "research"]
            ),

            # Section XII: Emerging Research 2025
            SectionInfo(
                id="12_emerging_research_2025",
                title="Emerging Research 2025",
                description="Latest AI research trends and emerging topics for 2025",
                path="12_Emerging_Research_2025/00_Overview.md",
                level=4,
                category="research",
                prerequisites=["all"],
                related_sections=["all"],
                estimated_time=420,
                topics_count=38,
                interactive_notebooks=6,
                last_updated="2024-10-05",
                tags=["research", "emerging", "2025", "latest", "trends"]
            ),

            # Section XIII: Advanced AI Security
            SectionInfo(
                id="13_advanced_security",
                title="Advanced AI Security and Defense",
                description="AI security, adversarial ML, and threat detection",
                path="13_Advanced_AI_Security/00_Overview.md",
                level=3,
                category="security",
                prerequisites=["01_foundational_ml", "07_ai_ethics_safety"],
                related_sections=["07_ai_ethics_safety", "23_aerospace_defense"],
                estimated_time=480,
                topics_count=44,
                interactive_notebooks=7,
                last_updated="2024-09-22",
                tags=["security", "adversarial", "defense", "threat_detection", "robustness"]
            ),

            # Section XIV: AI Business Enterprise
            SectionInfo(
                id="14_ai_business_enterprise",
                title="AI in Business and Enterprise",
                description="Enterprise AI strategy, architecture, and business applications",
                path="14_AI_Business_Enterprise/00_Overview.md",
                level=2,
                category="business",
                prerequisites=["08_ai_applications_industry"],
                related_sections=["08_ai_applications_industry", "10_technical_methodological"],
                estimated_time=440,
                topics_count=40,
                interactive_notebooks=6,
                last_updated="2024-09-26",
                tags=["business", "enterprise", "strategy", "architecture", "roi"]
            ),

            # Section XV: Specialized Applications
            SectionInfo(
                id="15_specialized_apps",
                title="Specialized AI Applications",
                description="Healthcare AI, robotics, creative AI, and autonomous systems",
                path="15_Specialized_Applications/00_Overview.md",
                level=3,
                category="applications",
                prerequisites=["04_computer_vision", "06_ai_agents"],
                related_sections=["04_computer_vision", "06_ai_agents", "21_agriculture_food", "22_smart_cities"],
                estimated_time=560,
                topics_count=52,
                interactive_notebooks=9,
                last_updated="2024-09-29",
                tags=["specialized", "healthcare", "robotics", "creative", "autonomous"]
            ),

            # Section XVI: Emerging AI Paradigms
            SectionInfo(
                id="16_emerging_paradigms",
                title="Emerging AI Paradigms",
                description="Edge AI, explainable AI, and scientific discovery AI",
                path="16_Emerging_AI_Paradigms/00_Overview.md",
                level=3,
                category="research",
                prerequisites=["01_foundational_ml", "07_ai_ethics_safety"],
                related_sections=["all"],
                estimated_time=460,
                topics_count=43,
                interactive_notebooks=7,
                last_updated="2024-10-03",
                tags=["edge_ai", "explainable", "scientific_discovery", "paradigms", "emerging"]
            ),

            # Section XVII: AI Social Good Impact
            SectionInfo(
                id="17_ai_social_good",
                title="AI for Social Good and Impact",
                description="Social impact applications and ethical AI development",
                path="17_AI_Social_Good_Impact.md",
                level=2,
                category="social",
                prerequisites=["07_ai_ethics_safety"],
                related_sections=["07_ai_ethics_safety", "08_ai_applications_industry", "21_agriculture_food"],
                estimated_time=380,
                topics_count=34,
                interactive_notebooks=5,
                last_updated="2024-09-24",
                tags=["social_good", "impact", "education", "healthcare_access", "environmental"]
            ),

            # Section XVIII: AI Policy Regulation
            SectionInfo(
                id="18_ai_policy_regulation",
                title="AI Policy and Regulation",
                description="Global governance, legal frameworks, and compliance",
                path="18_AI_Policy_and_Regulation.md",
                level=2,
                category="policy",
                prerequisites=["07_ai_ethics_safety"],
                related_sections=["07_ai_ethics_safety", "14_ai_business_enterprise", "25_ai_legal_regulatory"],
                estimated_time=400,
                topics_count=36,
                interactive_notebooks=4,
                last_updated="2024-09-27",
                tags=["policy", "regulation", "governance", "compliance", "legal"]
            ),

            # Section XIX: Human AI Collaboration
            SectionInfo(
                id="19_human_ai_collaboration",
                title="Human-AI Collaboration and Augmentation",
                description="Workforce augmentation and cognitive enhancement",
                path="19_Human_AI_Collaboration_and_Augmentation.md",
                level=3,
                category="human",
                prerequisites=["07_ai_ethics_safety"],
                related_sections=["17_ai_social_good", "20_entertainment_media"],
                estimated_time=420,
                topics_count=39,
                interactive_notebooks=6,
                last_updated="2024-10-02",
                tags=["collaboration", "augmentation", "workforce", "cognitive", "interfaces"]
            ),

            # Section XX: AI Entertainment Media
            SectionInfo(
                id="20_entertainment_media",
                title="AI in Entertainment and Media",
                description="Gaming, film, music, and creative content AI",
                path="20_AI_in_Entertainment_and_Media.md",
                level=2,
                category="creative",
                prerequisites=["05_generative_ai"],
                related_sections=["05_generative_ai", "19_human_ai_collaboration"],
                estimated_time=360,
                topics_count=33,
                interactive_notebooks=5,
                last_updated="2024-09-25",
                tags=["entertainment", "media", "gaming", "film", "music", "creative"]
            ),

            # Section XXI: AI Agriculture Food
            SectionInfo(
                id="21_agriculture_food",
                title="AI in Agriculture and Food Systems",
                description="Precision agriculture and food system optimization",
                path="21_AI_in_Agriculture_and_Food_Systems.md",
                level=2,
                category="industry",
                prerequisites=["01_foundational_ml", "04_computer_vision"],
                related_sections=["17_ai_social_good", "22_smart_cities"],
                estimated_time=380,
                topics_count=35,
                interactive_notebooks=6,
                last_updated="2024-09-23",
                tags=["agriculture", "food", "precision_farming", "sustainability", "monitoring"]
            ),

            # Section XXII: AI Smart Cities
            SectionInfo(
                id="22_smart_cities",
                title="AI for Smart Cities and Infrastructure",
                description="Urban intelligence and infrastructure AI",
                path="22_AI_for_Smart_Cities_and_Infrastructure.md",
                level=2,
                category="industry",
                prerequisites=["01_foundational_ml", "06_ai_agents"],
                related_sections=["21_agriculture_food", "24_ai_energy_environment"],
                estimated_time=400,
                topics_count=37,
                interactive_notebooks=6,
                last_updated="2024-09-30",
                tags=["smart_cities", "infrastructure", "urban", "traffic", "utilities"]
            ),

            # Section XXIII: AI Aerospace Defense
            SectionInfo(
                id="23_aerospace_defense",
                title="AI in Aerospace and Defense",
                description="Aerospace AI, defense systems, and space exploration",
                path="23_AI_in_Aerospace_and_Defense.md",
                level=3,
                category="industry",
                prerequisites=["06_ai_agents", "13_advanced_security"],
                related_sections=["13_advanced_security", "24_ai_energy_environment"],
                estimated_time=440,
                topics_count=41,
                interactive_notebooks=7,
                last_updated="2024-10-01",
                tags=["aerospace", "defense", "space", "autonomous_flight", "military"]
            ),

            # Section XXIV: AI Energy Environment
            SectionInfo(
                id="24_ai_energy_environment",
                title="AI in Energy and Environment",
                description="Energy AI and environmental monitoring systems",
                path="24_AI_in_Energy_and_Climate.md",
                level=2,
                category="industry",
                prerequisites=["01_foundational_ml", "09_emerging_interdisciplinary"],
                related_sections=["22_smart_cities", "23_aerospace_defense"],
                estimated_time=420,
                topics_count=38,
                interactive_notebooks=6,
                last_updated="2024-10-04",
                tags=["energy", "environment", "climate", "smart_grids", "sustainability"]
            ),

            # Section XXV: AI Legal Regulatory
            SectionInfo(
                id="25_ai_legal_regulatory",
                title="AI in Legal and Regulatory Systems",
                description="Legal AI, compliance automation, and regulatory technology",
                path="25_AI_Legal_and_Regulatory_Systems.md",
                level=3,
                category="legal",
                prerequisites=["18_ai_policy_regulation"],
                related_sections=["18_ai_policy_regulation", "14_ai_business_enterprise"],
                estimated_time=400,
                topics_count=36,
                interactive_notebooks=5,
                last_updated="2024-10-03",
                tags=["legal", "regulatory", "compliance", "contracts", "intellectual_property"]
            )
        ]

        # Add sections to the dictionary
        for section in sections_data:
            self.sections[section.id] = section

    def _initialize_learning_paths(self):
        """Initialize predefined learning paths for different user types."""
        learning_paths_data = [
            # Beginner Path
            LearningPath(
                id="beginner_path",
                name="AI Fundamentals Beginner Path",
                description="Comprehensive introduction to AI for complete beginners",
                target_audience="Beginners with no prior AI experience",
                sections=[
                    "01_foundational_ml",
                    "02_advanced_dl",  # Basic architectures only
                    "07_ai_ethics_safety",
                    "08_ai_applications_industry"  # Overview only
                ],
                estimated_duration=40,
                difficulty_level=1,
                outcomes=[
                    "Understand fundamental ML concepts",
                    "Build basic neural networks",
                    "Recognize ethical considerations in AI",
                    "Identify real-world AI applications"
                ]
            ),

            # Intermediate Path
            LearningPath(
                id="intermediate_path",
                name="AI Practitioner Intermediate Path",
                description="Deep dive into core AI technologies and applications",
                target_audience="Developers with basic programming experience",
                sections=[
                    "01_foundational_ml",
                    "02_advanced_dl",
                    "03_nlp",
                    "04_computer_vision",
                    "05_generative_ai",
                    "06_ai_agents",
                    "07_ai_ethics_safety",
                    "10_technical_methodological"
                ],
                estimated_duration=80,
                difficulty_level=2,
                outcomes=[
                    "Implement advanced neural architectures",
                    "Build NLP and computer vision systems",
                    "Create generative AI applications",
                    "Develop autonomous agents",
                    "Deploy AI systems responsibly"
                ]
            ),

            # Advanced Path
            LearningPath(
                id="advanced_path",
                name="AI Expert Advanced Path",
                description="Advanced AI concepts and cutting-edge research",
                target_audience="AI practitioners and researchers",
                sections=[
                    "09_emerging_interdisciplinary",
                    "11_future_directions",
                    "12_emerging_research_2025",
                    "13_advanced_security",
                    "16_emerging_paradigms"
                ],
                estimated_duration=60,
                difficulty_level=3,
                outcomes=[
                    "Understand emerging AI paradigms",
                    "Analyze advanced AI research",
                    "Implement AI security measures",
                    "Evaluate future AI trends"
                ]
            ),

            # Industry Professional Path
            LearningPath(
                id="industry_path",
                name="AI Industry Professional Path",
                description="Business-focused AI applications and enterprise deployment",
                target_audience="Business professionals and industry practitioners",
                sections=[
                    "08_ai_applications_industry",
                    "14_ai_business_enterprise",
                    "18_ai_policy_regulation",
                    "17_ai_social_good",
                    "25_ai_legal_regulatory"
                ],
                estimated_duration=45,
                difficulty_level=2,
                outcomes=[
                    "Identify AI business opportunities",
                    "Plan enterprise AI strategy",
                    "Navigate AI regulations",
                    "Implement responsible AI practices"
                ]
            ),

            # Researcher Path
            LearningPath(
                id="researcher_path",
                name="AI Researcher Path",
                description="Comprehensive research-oriented AI learning",
                target_audience="Academic researchers and PhD students",
                sections=[
                    "01_foundational_ml",
                    "02_advanced_dl",
                    "09_emerging_interdisciplinary",
                    "11_future_directions",
                    "12_emerging_research_2025",
                    "16_emerging_paradigms",
                    "07_ai_ethics_safety"
                ],
                estimated_duration=120,
                difficulty_level=4,
                outcomes=[
                    "Master theoretical foundations",
                    "Contribute to AI research",
                    "Develop novel AI approaches",
                    "Address fundamental research challenges"
                ]
            ),

            # Practitioner Path
            LearningPath(
                id="practitioner_path",
                name="AI Practitioner Specialized Path",
                description="Hands-on AI development and deployment",
                target_audience="Software developers and engineers",
                sections=[
                    "01_foundational_ml",
                    "02_advanced_dl",
                    "03_nlp",
                    "04_computer_vision",
                    "05_generative_ai",
                    "06_ai_agents",
                    "10_technical_methodological",
                    "15_specialized_apps",
                    "21_agriculture_food",
                    "22_smart_cities",
                    "23_aerospace_defense",
                    "24_ai_energy_environment"
                ],
                estimated_duration=150,
                difficulty_level=3,
                outcomes=[
                    "Build production-ready AI systems",
                    "Specialize in domain-specific applications",
                    "Optimize AI performance and efficiency",
                    "Deploy AI at scale"
                ]
            )
        ]

        # Add learning paths to the dictionary
        for path in learning_paths_data:
            self.learning_paths[path.id] = path

    def _build_cross_references(self):
        """Build cross-reference mapping between related sections."""
        for section_id, section in self.sections.items():
            # Initialize cross-references for this section
            self.cross_references[section_id] = set()

            # Add explicitly related sections
            for related_id in section.related_sections:
                if related_id == "all":
                    # Add all sections as related
                    self.cross_references[section_id].update(
                        other_id for other_id in self.sections.keys()
                        if other_id != section_id
                    )
                elif related_id in self.sections:
                    self.cross_references[section_id].add(related_id)

            # Add bidirectional relationships
            for related_id in self.cross_references[section_id].copy():
                self.cross_references[related_id].add(section_id)

    def _build_knowledge_graph(self):
        """Build knowledge graph representing conceptual relationships."""
        # Define conceptual clusters
        conceptual_clusters = {
            "foundations": ["01_foundational_ml", "07_ai_ethics_safety"],
            "core_technologies": ["02_advanced_dl", "03_nlp", "04_computer_vision", "05_generative_ai"],
            "applications": ["08_ai_applications_industry", "15_specialized_apps", "20_entertainment_media"],
            "industry_specific": ["21_agriculture_food", "22_smart_cities", "23_aerospace_defense", "24_ai_energy_environment"],
            "business_legal": ["14_ai_business_enterprise", "18_ai_policy_regulation", "25_ai_legal_regulatory"],
            "research_advanced": ["09_emerging_interdisciplinary", "11_future_directions", "12_emerging_research_2025", "16_emerging_paradigms"],
            "human_social": ["17_ai_social_good", "19_human_ai_collaboration"],
            "systems_infrastructure": ["06_ai_agents", "10_technical_methodological", "13_advanced_security"]
        }

        # Build knowledge graph based on conceptual relationships
        for cluster_name, section_ids in conceptual_clusters.items():
            for section_id in section_ids:
                if section_id in self.sections:
                    self.knowledge_graph[section_id] = [
                        other_id for other_id in section_ids
                        if other_id != section_id and other_id in self.sections
                    ]

    def get_section_navigation(self, section_id: str) -> Dict:
        """Get comprehensive navigation information for a section."""
        if section_id not in self.sections:
            return {}

        section = self.sections[section_id]

        # Build navigation data
        navigation = {
            "section": asdict(section),
            "prerequisites": [
                asdict(self.sections[prereq_id])
                for prereq_id in section.prerequisites
                if prereq_id in self.sections
            ],
            "related_sections": [
                asdict(self.sections[related_id])
                for related_id in self.cross_references.get(section_id, set())
                if related_id in self.sections
            ],
            "knowledge_graph": [
                asdict(self.sections[graph_id])
                for graph_id in self.knowledge_graph.get(section_id, [])
                if graph_id in self.sections
            ],
            "breadcrumb": self._get_breadcrumb(section_id),
            "quick_actions": self._get_quick_actions(section_id),
            "see_also": self._get_see_also_suggestions(section_id)
        }

        return navigation

    def _get_breadcrumb(self, section_id: str) -> List[Dict]:
        """Generate breadcrumb navigation for a section."""
        section = self.sections[section_id]
        breadcrumb = [
            {"title": "AI Documentation", "path": "00_Overview.md"},
        ]

        # Add category level
        category_map = {
            "foundations": "Foundations",
            "core": "Core Technologies",
            "advanced": "Advanced Topics",
            "applications": "Applications",
            "research": "Research",
            "business": "Business & Enterprise",
            "technical": "Technical & Infrastructure",
            "security": "Security & Defense",
            "social": "Social & Ethical",
            "policy": "Policy & Regulation",
            "human": "Human-AI Interaction",
            "creative": "Creative & Entertainment",
            "industry": "Industry Applications",
            "legal": "Legal & Regulatory"
        }

        if section.category in category_map:
            breadcrumb.append({
                "title": category_map[section.category],
                "path": f"00_Overview.md#{section.category}"
            })

        # Add current section
        breadcrumb.append({
            "title": section.title,
            "path": section.path
        })

        return breadcrumb

    def _get_quick_actions(self, section_id: str) -> List[Dict]:
        """Get quick actions for a section."""
        section = self.sections[section_id]
        actions = []

        # Add to favorites
        actions.append({
            "action": "add_favorite",
            "label": "Add to Favorites",
            "icon": "star"
        })

        # Mark as completed
        actions.append({
            "action": "mark_completed",
            "label": "Mark as Completed",
            "icon": "check"
        })

        # Share section
        actions.append({
            "action": "share",
            "label": "Share Section",
            "icon": "share"
        })

        # Download PDF (if available)
        actions.append({
            "action": "download_pdf",
            "label": "Download PDF",
            "icon": "download"
        })

        # View interactive notebooks
        if section.interactive_notebooks > 0:
            actions.append({
                "action": "open_notebooks",
                "label": f"Open {section.interactive_notebooks} Interactive Notebooks",
                "icon": "code"
            })

        return actions

    def _get_see_also_suggestions(self, section_id: str) -> List[Dict]:
        """Get "See Also" suggestions for deeper learning."""
        section = self.sections[section_id]
        suggestions = []

        # Suggest next sections based on prerequisites
        for other_id, other_section in self.sections.items():
            if section_id in other_section.prerequisites and other_id not in section.related_sections:
                suggestions.append({
                    "type": "next_step",
                    "title": f"Next: {other_section.title}",
                    "description": f"Build on your knowledge of {section.title}",
                    "path": other_section.path
                })

        # Suggest sections with similar tags
        for other_id, other_section in self.sections.items():
            if other_id != section_id and other_id not in section.related_sections:
                common_tags = set(section.tags) & set(other_section.tags)
                if len(common_tags) >= 2:  # At least 2 common tags
                    suggestions.append({
                        "type": "similar_topic",
                        "title": other_section.title,
                        "description": f"Similar topics: {', '.join(list(common_tags)[:3])}",
                        "path": other_section.path
                    })

        return suggestions[:5]  # Limit to top 5 suggestions

    def get_learning_path_navigation(self, path_id: str, current_position: int = 0) -> Dict:
        """Get navigation for a specific learning path."""
        if path_id not in self.learning_paths:
            return {}

        path = self.learning_paths[path_id]

        # Build navigation data
        navigation = {
            "path": asdict(path),
            "sections": [],
            "progress": {
                "current_section": path.sections[current_position] if current_position < len(path.sections) else None,
                "completed_sections": path.sections[:current_position],
                "remaining_sections": path.sections[current_position + 1:],
                "progress_percent": (current_position / len(path.sections)) * 100
            },
            "next_actions": self._get_path_actions(path_id, current_position)
        }

        # Add section details
        for i, section_id in enumerate(path.sections):
            if section_id in self.sections:
                section = self.sections[section_id]
                navigation["sections"].append({
                    "section": asdict(section),
                    "position": i + 1,
                    "status": "completed" if i < current_position else "current" if i == current_position else "upcoming",
                    "estimated_time": section.estimated_time
                })

        return navigation

    def _get_path_actions(self, path_id: str, current_position: int) -> List[Dict]:
        """Get available actions for current position in learning path."""
        actions = []
        path = self.learning_paths[path_id]

        if current_position < len(path.sections):
            # Continue to next section
            actions.append({
                "action": "continue_learning",
                "label": "Continue to Next Section",
                "icon": "arrow_forward"
            })
        else:
            # Path completed
            actions.append({
                "action": "complete_path",
                "label": "Mark Path as Completed",
                "icon": "trophy"
            })
            actions.append({
                "action": "explore_paths",
                "label": "Explore Other Learning Paths",
                "icon": "explore"
            })

        # Review progress
        if current_position > 0:
            actions.append({
                "action": "review_progress",
                "label": "Review Progress",
                "icon": "assessment"
            })

        return actions

    def search_sections(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search sections based on query and filters."""
        results = []
        query_lower = query.lower()

        for section_id, section in self.sections.items():
            # Text search
            score = 0
            if query_lower in section.title.lower():
                score += 10
            if query_lower in section.description.lower():
                score += 5
            for tag in section.tags:
                if query_lower in tag.lower():
                    score += 3

            # Apply filters
            if filters:
                if "level" in filters and section.level != filters["level"]:
                    continue
                if "category" in filters and section.category != filters["category"]:
                    continue
                if "tags" in filters:
                    if not any(tag in section.tags for tag in filters["tags"]):
                        continue

            if score > 0:
                results.append({
                    "section": asdict(section),
                    "relevance_score": score,
                    "match_reasons": self._get_match_reasons(section, query_lower)
                })

        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    def _get_match_reasons(self, section: SectionInfo, query: str) -> List[str]:
        """Get reasons why a section matches the search query."""
        reasons = []

        if query in section.title.lower():
            reasons.append("Title match")

        if query in section.description.lower():
            reasons.append("Description match")

        matching_tags = [tag for tag in section.tags if query in tag.lower()]
        if matching_tags:
            reasons.append(f"Tags: {', '.join(matching_tags[:2])}")

        return reasons

    def get_quick_jump_options(self) -> Dict:
        """Get quick jump navigation options."""
        return {
            "categories": {
                "Foundations": ["01_foundational_ml", "07_ai_ethics_safety"],
                "Core Technologies": ["02_advanced_dl", "03_nlp", "04_computer_vision", "05_generative_ai"],
                "Applications": ["08_ai_applications_industry", "15_specialized_apps"],
                "Research": ["09_emerging_interdisciplinary", "11_future_directions", "12_emerging_research_2025"],
                "Industry": ["14_ai_business_enterprise", "21_agriculture_food", "22_smart_cities", "23_aerospace_defense", "24_ai_energy_environment"],
                "Policy & Legal": ["18_ai_policy_regulation", "25_ai_legal_regulatory"]
            },
            "difficulty_levels": {
                "Beginner": [s for s in self.sections.values() if s.level == 1],
                "Intermediate": [s for s in self.sections.values() if s.level == 2],
                "Advanced": [s for s in self.sections.values() if s.level == 3],
                "Research": [s for s in self.sections.values() if s.level == 4]
            },
            "most_accessed": [
                "01_foundational_ml", "02_advanced_dl", "03_nlp", "05_generative_ai", "07_ai_ethics_safety"
            ],
            "interactive_notebooks": [
                s for s in self.sections.values() if s.interactive_notebooks > 0
            ]
        }

    def get_keyboard_shortcuts(self) -> Dict[str, Dict]:
        """Get available keyboard shortcuts."""
        return {
            "navigation": {
                "Alt + ←": "Go to previous section",
                "Alt + →": "Go to next section",
                "Alt + ↑": "Go to parent section",
                "Alt + ↓": "Go to subsection",
                "Ctrl + K": "Quick search",
                "Ctrl + /": "Show keyboard shortcuts",
                "Ctrl + H": "Go to home",
                "Ctrl + B": "Toggle sidebar",
                "F1": "Show help"
            },
            "content": {
                "Ctrl + F": "Find in current document",
                "Ctrl + G": "Find next",
                "Ctrl + Shift + G": "Find previous",
                "Ctrl + +": "Increase font size",
                "Ctrl + -": "Decrease font size",
                "Ctrl + 0": "Reset font size"
            },
            "learning": {
                "Ctrl + P": "Show learning paths",
                "Ctrl + T": "Toggle dark mode",
                "Ctrl + L": "Show table of contents",
                "Ctrl + M": "Add bookmark",
                "Ctrl + S": "Save progress"
            }
        }

    def export_navigation_data(self, format: str = "json") -> str:
        """Export navigation data in specified format."""
        data = {
            "sections": {sid: asdict(s) for sid, s in self.sections.items()},
            "learning_paths": {pid: asdict(p) for pid, p in self.learning_paths.items()},
            "cross_references": {k: list(v) for k, v in self.cross_references.items()},
            "knowledge_graph": self.knowledge_graph,
            "generated_at": datetime.now().isoformat()
        }

        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "python":
            return f"# AI Documentation Navigation Data\nnavigation_data = {data}"
        else:
            raise ValueError(f"Unsupported format: {format}")


def main():
    """Demo the navigation system."""
    navigator = AIDocumentationNavigator()

    # Example usage
    print("=== AI Documentation Navigation System ===\n")

    # Show quick jump options
    print("Quick Jump Options:")
    quick_jump = navigator.get_quick_jump_options()
    for category, sections in quick_jump["categories"].items():
        print(f"  {category}: {len(sections)} sections")

    # Example section navigation
    print("\n=== Example: Deep Learning Section Navigation ===")
    nav_data = navigator.get_section_navigation("02_advanced_dl")
    print(f"Section: {nav_data['section']['title']}")
    print(f"Related Sections: {len(nav_data['related_sections'])}")
    print(f"Breadcrumb: {' → '.join([b['title'] for b in nav_data['breadcrumb']])}")

    # Example learning path
    print("\n=== Example: Beginner Learning Path ===")
    path_nav = navigator.get_learning_path_navigation("beginner_path", 1)
    print(f"Path: {path_nav['path']['name']}")
    print(f"Progress: {path_nav['progress']['progress_percent']:.1f}%")
    print(f"Current Section: {path_nav['progress']['current_section']}")

    # Example search
    print("\n=== Example Search: 'ethics' ===")
    search_results = navigator.search_sections("ethics")
    for result in search_results[:3]:
        print(f"Found: {result['section']['title']} (Score: {result['relevance_score']})")

    print("\nNavigation system initialized successfully!")


if __name__ == "__main__":
    main()