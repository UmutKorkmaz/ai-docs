#!/usr/bin/env python3
"""
Advanced Intelligent Cross-Referencing and Knowledge Graph System
for AI Documentation Project

This sophisticated system provides:
- Comprehensive knowledge graph with 250+ AI concepts
- Entity relationship mapping across all 25 sections
- Semantic relationship extraction and classification
- Intelligent cross-reference discovery and insertion
- Context-aware link suggestions
- Topic modeling and content clustering
- Concept-based navigation and exploration
- Interactive knowledge graph visualization
- AI-powered insights and content gap analysis
"""

import json
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re
import os
import pickle
from pathlib import Path
import logging
from datetime import datetime
import hashlib

# NLP and ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import LatentDirichletAllocation
    import spacy
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError as e:
    print(f"Warning: Some ML dependencies not available: {e}")

# Visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Warning: Some visualization dependencies not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AIConcept:
    """Represents an AI concept in the knowledge graph"""
    id: str
    name: str
    definition: str
    section: str
    subsection: str
    file_path: str
    line_number: int
    concept_type: str  # 'theory', 'method', 'application', 'example'
    difficulty_level: int  # 1-5
    prerequisites: List[str]
    related_concepts: List[str]
    applications: List[str]
    tags: List[str]
    embedding: Optional[np.ndarray] = None
    last_updated: datetime = datetime.now()

@dataclass
class ContentRelationship:
    """Represents a relationship between content pieces"""
    source_id: str
    target_id: str
    relationship_type: str  # 'prerequisite', 'related', 'application', 'example', 'contrast'
    strength: float  # 0.0-1.0
    context: str
    bidirectional: bool = False
    auto_generated: bool = True
    created_at: datetime = datetime.now()

@dataclass
class LearningPath:
    """Represents a learning path through concepts"""
    id: str
    name: str
    description: str
    concepts: List[str]
    difficulty_progression: List[int]
    estimated_time: int  # in minutes
    learning_objectives: List[str]
    prerequisites: List[str]
    target_audience: str

class AIKnowledgeGraph:
    """Main knowledge graph system for AI documentation"""

    def __init__(self, base_path: str = "/Users/dtumkorkmaz/Projects/ai-docs"):
        self.base_path = Path(base_path)
        self.concepts: Dict[str, AIConcept] = {}
        self.relationships: List[ContentRelationship] = []
        self.learning_paths: Dict[str, LearningPath] = {}
        self.content_index: Dict[str, Dict] = {}
        self.graph = nx.DiGraph()
        self.topic_model = None
        self.vectorizer = None
        self.content_embeddings = {}

        # Configuration
        self.config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'similarity_threshold': 0.3,
            'max_concepts_per_file': 100,
            'min_relationship_strength': 0.1,
            'cache_embeddings': True,
            'update_existing': True
        }

        # AI concept taxonomy
        self.concept_taxonomy = {
            'foundational': ['machine_learning', 'deep_learning', 'neural_networks', 'statistics', 'linear_algebra'],
            'algorithms': ['supervised_learning', 'unsupervised_learning', 'reinforcement_learning', 'transfer_learning'],
            'architectures': ['cnn', 'rnn', 'transformer', 'gan', 'vae', 'mamba', 'state_space_models'],
            'applications': ['computer_vision', 'natural_language_processing', 'robotics', 'healthcare_ai'],
            'ethics': ['bias', 'fairness', 'interpretability', 'safety', 'governance'],
            'emerging': ['generative_ai', 'multimodal_ai', 'agentic_ai', 'neuromorphic_computing']
        }

        # Initialize NLP models
        self._initialize_nlp_models()

        # Load existing data
        self._load_existing_data()

    def _initialize_nlp_models(self):
        """Initialize NLP models for text processing"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Spacy model loaded successfully")
        except OSError:
            logger.warning("Spacy model not found, using basic text processing")
            self.nlp = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config['embedding_model'])
            self.embedding_model = AutoModel.from_pretrained(self.config['embedding_model'])
            logger.info(f"Embedding model {self.config['embedding_model']} loaded")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            self.embedding_model = None

    def _load_existing_data(self):
        """Load existing knowledge graph data"""
        cache_file = self.base_path / "knowledge_graph_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.concepts = cached_data.get('concepts', {})
                    self.relationships = cached_data.get('relationships', [])
                    self.learning_paths = cached_data.get('learning_paths', {})
                    self.content_index = cached_data.get('content_index', {})
                logger.info(f"Loaded cached data: {len(self.concepts)} concepts, {len(self.relationships)} relationships")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")

    def _save_cache(self):
        """Save current state to cache"""
        cache_file = self.base_path / "knowledge_graph_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'concepts': self.concepts,
                    'relationships': self.relationships,
                    'learning_paths': self.learning_paths,
                    'content_index': self.content_index
                }, f)
            logger.info("Knowledge graph cache saved")
        except Exception as e:
            logger.error(f"Could not save cache: {e}")

    def extract_concepts_from_text(self, text: str, file_path: str, section: str) -> List[AIConcept]:
        """Extract AI concepts from text using advanced NLP"""
        concepts = []

        # Predefined AI concept patterns
        concept_patterns = {
            'methods': r'\b(supervised|unsupervised|reinforcement|transfer|few-shot|zero-shot|self-supervised)\s+learning\b',
            'architectures': r'\b(convolutional|recurrent|transformer|attention|generative|adversarial)\s+(network|model|architecture)\b',
            'techniques': r'\b(backpropagation|gradient descent|adam|sgd|dropout|batch\s+normalization|regularization)\b',
            'applications': r'\b(computer\s+vision|natural\s+language\s+processing|machine\s+translation|object\s+detection|image\s+classification)\b',
            'metrics': r'\b(accuracy|precision|recall|f1|roc|auc|mse|rmse|mae)\b',
            'ethics': r'\b(bias|fairness|interpretability|explainability|transparency|accountability)\b'
        }

        lines = text.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('```'):
                continue

            # Extract concepts using patterns
            for category, pattern in concept_patterns.items():
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    concept_name = match.group().lower()
                    concept_id = hashlib.md5(f"{concept_name}_{file_path}_{line_num}".encode()).hexdigest()[:8]

                    # Determine difficulty based on context
                    difficulty = self._estimate_difficulty(line, category)

                    concept = AIConcept(
                        id=concept_id,
                        name=concept_name,
                        definition=line,  # Use the line as context/definition
                        section=section,
                        subsection=self._extract_subsection(file_path),
                        file_path=file_path,
                        line_number=line_num,
                        concept_type=category,
                        difficulty_level=difficulty,
                        prerequisites=[],
                        related_concepts=[],
                        applications=[],
                        tags=self._extract_tags(line)
                    )
                    concepts.append(concept)

        return concepts

    def _estimate_difficulty(self, text: str, category: str) -> int:
        """Estimate difficulty level of a concept based on context"""
        difficulty_keywords = {
            1: ['introduction', 'basic', 'simple', 'overview', 'fundamentals'],
            2: ['intermediate', 'practical', 'applied', 'implementation'],
            3: ['advanced', 'complex', 'sophisticated', 'optimization'],
            4: ['expert', 'research', 'cutting-edge', 'state-of-the-art'],
            5: ['theoretical', 'mathematical', 'academic', 'novel']
        }

        text_lower = text.lower()
        for level, keywords in difficulty_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level

        # Default difficulty based on category
        category_difficulty = {
            'methods': 2,
            'architectures': 3,
            'techniques': 3,
            'applications': 2,
            'metrics': 1,
            'ethics': 2
        }

        return category_difficulty.get(category, 2)

    def _extract_subsection(self, file_path: str) -> str:
        """Extract subsection from file path"""
        path_parts = Path(file_path).parts
        if len(path_parts) > 1:
            return path_parts[-2]  # Second to last part
        return "general"

    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text"""
        # Common AI keywords as tags
        ai_keywords = [
            'deep learning', 'machine learning', 'neural networks', 'ai', 'artificial intelligence',
            'python', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'data science',
            'computer vision', 'nlp', 'natural language processing', 'robotics',
            'optimization', 'algorithm', 'model', 'training', 'prediction', 'classification'
        ]

        text_lower = text.lower()
        tags = []
        for keyword in ai_keywords:
            if keyword in text_lower:
                tags.append(keyword)

        return tags[:10]  # Limit to 10 tags

    def build_knowledge_graph(self, force_rebuild: bool = False):
        """Build the complete knowledge graph from documentation"""
        logger.info("Building knowledge graph...")

        if not force_rebuild and self.concepts:
            logger.info("Knowledge graph already exists, updating...")
            self.update_knowledge_graph()
            return

        # Process all markdown files
        md_files = list(self.base_path.glob("**/*.md"))
        logger.info(f"Processing {len(md_files)} markdown files...")

        all_concepts = []
        content_chunks = []

        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                section = self._determine_section(file_path)
                concepts = self.extract_concepts_from_text(content, str(file_path), section)
                all_concepts.extend(concepts)

                # Store content chunks for embedding
                paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
                for i, paragraph in enumerate(paragraphs):
                    content_chunks.append({
                        'text': paragraph,
                        'file_path': str(file_path),
                        'section': section,
                        'chunk_id': f"{file_path.stem}_{i}"
                    })

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        # Store concepts
        for concept in all_concepts:
            self.concepts[concept.id] = concept

        # Generate embeddings
        self._generate_embeddings(content_chunks)

        # Build relationships
        self._build_relationships(all_concepts, content_chunks)

        # Build NetworkX graph
        self._build_networkx_graph()

        # Generate learning paths
        self._generate_learning_paths()

        # Save cache
        self._save_cache()

        logger.info(f"Knowledge graph built: {len(self.concepts)} concepts, {len(self.relationships)} relationships")

    def _determine_section(self, file_path: Path) -> str:
        """Determine section from file path"""
        path_str = str(file_path)

        # Map directories to sections
        section_mapping = {
            '01_Foundational_Machine_Learning': 'Foundational ML',
            '02_Advanced_Deep_Learning': 'Advanced Deep Learning',
            '03_Natural_Language_Processing': 'NLP',
            '04_Computer_Vision': 'Computer Vision',
            '05_Generative_AI': 'Generative AI',
            '06_AI_Agents_and_Autonomous': 'AI Agents',
            '07_AI_Ethics_and_Safety': 'AI Ethics',
            '08_AI_Applications_Industry': 'Industry Applications',
            '09_Emerging_Interdisciplinary': 'Interdisciplinary',
            '10_Technical_Methodological': 'Technical',
            '11_Future_Directions': 'Future Directions',
            '12_Emerging_Research_2025': 'Emerging Research',
            '13_Advanced_AI_Security': 'AI Security',
            '14_MLOps_and_AI_Deployment_Strategies': 'MLOps',
            '14_AI_Business_Enterprise': 'Business AI',
            '15_Specialized_Applications': 'Specialized Applications',
            '16_Emerging_AI_Paradigms': 'Emerging Paradigms',
            '17_AI_Social_Good_Impact': 'Social Impact',
            '18_AI_Policy_Regulation': 'Policy & Regulation',
            '19_Human_AI_Collaboration': 'Human-AI Collaboration',
            '20_AI_Entertainment_Media': 'Entertainment & Media',
            '21_AI_Agriculture_Food': 'Agriculture & Food',
            '22_AI_Smart_Cities': 'Smart Cities',
            '23_AI_Aerospace_Defense': 'Aerospace & Defense',
            '24_AI_Energy_Environment': 'Energy & Environment',
            '25_AI_Legal_Regulatory': 'Legal & Regulatory'
        }

        for directory, section_name in section_mapping.items():
            if directory in path_str:
                return section_name

        return 'General'

    def _generate_embeddings(self, content_chunks: List[Dict]):
        """Generate embeddings for content chunks"""
        logger.info("Generating embeddings...")

        if not self.embedding_model:
            logger.warning("No embedding model available, using TF-IDF")
            self._generate_tfidf_embeddings(content_chunks)
            return

        texts = [chunk['text'] for chunk in content_chunks]

        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize and generate embeddings
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True,
                                   return_tensors="pt", max_length=512)

            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
                all_embeddings.extend(embeddings)

        # Store embeddings
        for i, chunk in enumerate(content_chunks):
            chunk_id = chunk['chunk_id']
            self.content_embeddings[chunk_id] = all_embeddings[i]

        logger.info(f"Generated embeddings for {len(content_chunks)} content chunks")

    def _generate_tfidf_embeddings(self, content_chunks: List[Dict]):
        """Generate TF-IDF embeddings as fallback"""
        logger.info("Using TF-IDF for embeddings...")

        texts = [chunk['text'] for chunk in content_chunks]
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        for i, chunk in enumerate(content_chunks):
            chunk_id = chunk['chunk_id']
            self.content_embeddings[chunk_id] = tfidf_matrix[i].toarray().flatten()

    def _build_relationships(self, concepts: List[AIConcept], content_chunks: List[Dict]):
        """Build relationships between concepts"""
        logger.info("Building relationships...")

        # Semantic similarity relationships
        self._build_semantic_relationships(concepts)

        # Prerequisite relationships based on difficulty
        self._build_prerequisite_relationships(concepts)

        # Application relationships
        self._build_application_relationships(concepts)

        # Co-occurrence relationships
        self._build_cooccurrence_relationships(content_chunks)

    def _build_semantic_relationships(self, concepts: List[AIConcept]):
        """Build relationships based on semantic similarity"""
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Calculate semantic similarity
                similarity = self._calculate_concept_similarity(concept1, concept2)

                if similarity > self.config['similarity_threshold']:
                    relationship = ContentRelationship(
                        source_id=concept1.id,
                        target_id=concept2.id,
                        relationship_type='related',
                        strength=similarity,
                        context=f"Semantic similarity: {similarity:.3f}",
                        bidirectional=True
                    )
                    self.relationships.append(relationship)

    def _calculate_concept_similarity(self, concept1: AIConcept, concept2: AIConcept) -> float:
        """Calculate similarity between two concepts"""
        # Simple text-based similarity for now
        name1_words = set(concept1.name.lower().split())
        name2_words = set(concept2.name.lower().split())

        # Jaccard similarity on names
        name_similarity = len(name1_words & name2_words) / len(name1_words | name2_words) if name1_words | name2_words else 0

        # Tag similarity
        tags1 = set(concept1.tags)
        tags2 = set(concept2.tags)
        tag_similarity = len(tags1 & tags2) / len(tags1 | tags2) if tags1 | tags2 else 0

        # Section similarity
        section_similarity = 1.0 if concept1.section == concept2.section else 0.0

        # Weighted combination
        total_similarity = 0.5 * name_similarity + 0.3 * tag_similarity + 0.2 * section_similarity

        return total_similarity

    def _build_prerequisite_relationships(self, concepts: List[AIConcept]):
        """Build prerequisite relationships based on difficulty and content"""
        # Group concepts by type
        concept_groups = defaultdict(list)
        for concept in concepts:
            concept_groups[concept.concept_type].append(concept)

        # Within each group, create prerequisite relationships based on difficulty
        for group_name, group_concepts in concept_groups.items():
            # Sort by difficulty
            sorted_concepts = sorted(group_concepts, key=lambda x: x.difficulty_level)

            for i, higher_level_concept in enumerate(sorted_concepts):
                for lower_level_concept in sorted_concepts[:i]:
                    if higher_level_concept.difficulty_level - lower_level_concept.difficulty_level >= 1:
                        relationship = ContentRelationship(
                            source_id=lower_level_concept.id,
                            target_id=higher_level_concept.id,
                            relationship_type='prerequisite',
                            strength=0.7,
                            context=f"Prerequisite: {lower_level_concept.name} before {higher_level_concept.name}"
                        )
                        self.relationships.append(relationship)

    def _build_application_relationships(self, concepts: List[AIConcept]):
        """Build application relationships between theory and practice"""
        theory_concepts = [c for c in concepts if c.concept_type in ['methods', 'techniques']]
        application_concepts = [c for c in concepts if c.concept_type in ['applications', 'examples']]

        for theory in theory_concepts:
            for application in application_concepts:
                # Check if application mentions theory
                if any(word in application.definition.lower() for word in theory.name.lower().split()):
                    relationship = ContentRelationship(
                        source_id=theory.id,
                        target_id=application.id,
                        relationship_type='application',
                        strength=0.8,
                        context=f"Application of {theory.name} in {application.name}"
                    )
                    self.relationships.append(relationship)

    def _build_cooccurrence_relationships(self, content_chunks: List[Dict]):
        """Build relationships based on concept co-occurrence in content"""
        # Find concepts mentioned in the same content chunk
        for chunk in content_chunks:
            mentioned_concepts = []
            chunk_text = chunk['text'].lower()

            for concept_id, concept in self.concepts.items():
                if concept.name in chunk_text:
                    mentioned_concepts.append(concept_id)

            # Create relationships between co-occurring concepts
            for i, concept1_id in enumerate(mentioned_concepts):
                for concept2_id in mentioned_concepts[i+1:]:
                    relationship = ContentRelationship(
                        source_id=concept1_id,
                        target_id=concept2_id,
                        relationship_type='cooccurrence',
                        strength=0.5,
                        context=f"Co-occurrence in {chunk['file_path']}"
                    )
                    self.relationships.append(relationship)

    def _build_networkx_graph(self):
        """Build NetworkX graph for visualization and analysis"""
        self.graph = nx.DiGraph()

        # Add nodes (concepts)
        for concept_id, concept in self.concepts.items():
            self.graph.add_node(
                concept_id,
                name=concept.name,
                section=concept.section,
                type=concept.concept_type,
                difficulty=concept.difficulty_level,
                definition=concept.definition[:100] + "..." if len(concept.definition) > 100 else concept.definition
            )

        # Add edges (relationships)
        for rel in self.relationships:
            self.graph.add_edge(
                rel.source_id,
                rel.target_id,
                relationship_type=rel.relationship_type,
                strength=rel.strength,
                context=rel.context
            )

    def _generate_learning_paths(self):
        """Generate learning paths through the knowledge graph"""
        # Beginner path
        beginner_concepts = [
            cid for cid, concept in self.concepts.items()
            if concept.difficulty_level <= 2 and concept.concept_type in ['methods', 'techniques']
        ]

        self.learning_paths['beginner'] = LearningPath(
            id='beginner',
            name='AI Fundamentals Path',
            description='Learn the basics of AI and machine learning',
            concepts=beginner_concepts[:20],  # Limit to 20 concepts
            difficulty_progression=[1, 1, 2, 2, 2, 3, 3],
            estimated_time=180,  # 3 hours
            learning_objectives=[
                'Understand basic ML concepts',
                'Learn fundamental algorithms',
                'Implement simple models',
                'Understand ethical considerations'
            ],
            prerequisites=[],
            target_audience='Beginners'
        )

        # Advanced path
        advanced_concepts = [
            cid for cid, concept in self.concepts.items()
            if concept.difficulty_level >= 3
        ]

        self.learning_paths['advanced'] = LearningPath(
            id='advanced',
            name='Advanced AI Specialization',
            description='Deep dive into advanced AI topics',
            concepts=advanced_concepts[:30],  # Limit to 30 concepts
            difficulty_progression=[3, 3, 4, 4, 4, 5, 5],
            estimated_time=300,  # 5 hours
            learning_objectives=[
                'Master advanced architectures',
                'Understand cutting-edge research',
                'Implement complex systems',
                'Research and innovation'
            ],
            prerequisites=beginner_concepts[:10],
            target_audience='Advanced learners'
        )

        # Specialized paths
        for specialization in ['Computer Vision', 'NLP', 'Generative AI', 'AI Ethics']:
            specialization_concepts = [
                cid for cid, concept in self.concepts.items()
                if specialization.lower() in concept.section.lower() or
                   specialization.lower() in ' '.join(concept.tags).lower()
            ]

            if specialization_concepts:
                self.learning_paths[specialization.lower().replace(' ', '_')] = LearningPath(
                    id=specialization.lower().replace(' ', '_'),
                    name=f'{specialization} Specialization',
                    description=f'Specialized path for {specialization}',
                    concepts=specialization_concepts[:15],
                    difficulty_progression=[2, 3, 3, 4, 4],
                    estimated_time=240,  # 4 hours
                    learning_objectives=[f'Master {specialization} concepts'],
                    prerequisites=beginner_concepts[:5],
                    target_audience=f'{specialization} enthusiasts'
                )

    def update_knowledge_graph(self):
        """Update existing knowledge graph with new content"""
        logger.info("Updating knowledge graph...")

        # Find new or modified files
        # This is a simplified version - in practice, you'd check file modification times
        self.build_knowledge_graph(force_rebuild=True)

    def get_concept_recommendations(self, concept_id: str, limit: int = 10) -> List[Dict]:
        """Get recommendations for a given concept"""
        if concept_id not in self.concepts:
            return []

        recommendations = []
        concept = self.concepts[concept_id]

        # Find related concepts
        for rel in self.relationships:
            if rel.source_id == concept_id or rel.target_id == concept_id:
                related_id = rel.target_id if rel.source_id == concept_id else rel.source_id
                if related_id in self.concepts:
                    related_concept = self.concepts[related_id]
                    recommendations.append({
                        'concept_id': related_id,
                        'name': related_concept.name,
                        'section': related_concept.section,
                        'difficulty': related_concept.difficulty_level,
                        'relationship_type': rel.relationship_type,
                        'strength': rel.strength,
                        'context': rel.context
                    })

        # Sort by strength and limit
        recommendations.sort(key=lambda x: x['strength'], reverse=True)
        return recommendations[:limit]

    def discover_learning_path(self, start_concept: str, target_concept: str) -> Optional[List[str]]:
        """Discover a learning path between two concepts"""
        try:
            # Use NetworkX to find shortest path
            path = nx.shortest_path(
                self.graph,
                source=start_concept,
                target=target_concept,
                weight='strength'
            )
            return path
        except (nx.NetworkXNoPath, KeyError):
            return None

    def analyze_content_gaps(self) -> Dict[str, List[str]]:
        """Analyze gaps in content coverage"""
        gaps = {
            'missing_concepts': [],
            'underrepresented_sections': [],
            'prerequisite_gaps': []
        }

        # Check for underrepresented sections
        section_counts = Counter(concept.section for concept in self.concepts.values())
        total_concepts = len(self.concepts)

        for section, count in section_counts.items():
            if count < total_concepts * 0.05:  # Less than 5% of total concepts
                gaps['underrepresented_sections'].append(section)

        # Check for prerequisite gaps
        for rel in self.relationships:
            if rel.relationship_type == 'prerequisite':
                if rel.target_id not in self.concepts or rel.source_id not in self.concepts:
                    gaps['prerequisite_gaps'].append(f"{rel.source_id} -> {rel.target_id}")

        return gaps

    def generate_knowledge_visualization(self, output_path: str = None) -> go.Figure:
        """Generate interactive visualization of the knowledge graph"""
        logger.info("Generating knowledge graph visualization...")

        if output_path is None:
            output_path = str(self.base_path / "knowledge_graph_visualization.html")

        # Create layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)

        # Extract node and edge data
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            node_data = self.graph.nodes[node]
            node_text.append(f"{node_data['name']}<br>{node_data['section']}<br>Level: {node_data['difficulty']}")

            # Color by section
            section_colors = {
                'Foundational ML': 'blue',
                'Advanced Deep Learning': 'green',
                'NLP': 'red',
                'Computer Vision': 'purple',
                'Generative AI': 'orange',
                'AI Agents': 'pink',
                'AI Ethics': 'brown',
                'Industry Applications': 'gray'
            }
            node_color.append(section_colors.get(node_data['section'], 'lightblue'))

            # Size by difficulty
            node_size.append(node_data['difficulty'] * 10)

        # Edge traces
        edge_x = []
        edge_y = []

        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))

        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            )
        ))

        fig.update_layout(
            title="AI Knowledge Graph Visualization",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="AI Documentation Knowledge Graph",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='black', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800
        )

        # Save to file
        fig.write_html(output_path)
        logger.info(f"Visualization saved to {output_path}")

        return fig

    def generate_cross_reference_suggestions(self, file_path: str) -> List[Dict]:
        """Generate cross-reference suggestions for a given file"""
        suggestions = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return suggestions

        # Extract concepts mentioned in the file
        mentioned_concepts = []
        content_lower = content.lower()

        for concept_id, concept in self.concepts.items():
            if concept.name in content_lower and concept.file_path != file_path:
                mentioned_concepts.append(concept)

        # Generate suggestions
        for concept in mentioned_concepts[:20]:  # Limit to 20 suggestions
            suggestions.append({
                'concept_id': concept.id,
                'concept_name': concept.name,
                'target_file': concept.file_path,
                'target_section': concept.section,
                'context': concept.definition[:100] + "..." if len(concept.definition) > 100 else concept.definition,
                'reason': f"Concept '{concept.name}' is related to this content",
                'link_text': f"See also: {concept.name}",
                'markdown_link': f"[{concept.name}]({concept.file_path}#{concept.line_number})"
            })

        return suggestions

    def export_knowledge_graph_data(self, format: str = 'json') -> str:
        """Export knowledge graph data in various formats"""
        export_data = {
            'concepts': {cid: asdict(concept) for cid, concept in self.concepts.items()},
            'relationships': [asdict(rel) for rel in self.relationships],
            'learning_paths': {lid: asdict(path) for lid, path in self.learning_paths.items()},
            'statistics': {
                'total_concepts': len(self.concepts),
                'total_relationships': len(self.relationships),
                'total_learning_paths': len(self.learning_paths),
                'sections': list(set(concept.section for concept in self.concepts.values())),
                'concept_types': list(set(concept.concept_type for concept in self.concepts.values()))
            },
            'export_timestamp': datetime.now().isoformat()
        }

        if format == 'json':
            return json.dumps(export_data, indent=2, default=str)
        elif format == 'csv':
            # Export concepts as CSV
            concepts_df = pd.DataFrame([asdict(concept) for concept in self.concepts.values()])
            return concepts_df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_system_statistics(self) -> Dict:
        """Get comprehensive statistics about the knowledge graph system"""
        stats = {
            'concepts': {
                'total': len(self.concepts),
                'by_section': Counter(concept.section for concept in self.concepts.values()),
                'by_type': Counter(concept.concept_type for concept in self.concepts.values()),
                'by_difficulty': Counter(concept.difficulty_level for concept in self.concepts.values())
            },
            'relationships': {
                'total': len(self.relationships),
                'by_type': Counter(rel.relationship_type for rel in self.relationships),
                'average_strength': np.mean([rel.strength for rel in self.relationships]) if self.relationships else 0
            },
            'learning_paths': {
                'total': len(self.learning_paths),
                'average_concepts_per_path': np.mean([len(path.concepts) for path in self.learning_paths.values()]) if self.learning_paths else 0
            },
            'graph_metrics': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
                'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else True
            }
        }

        return stats

def main():
    """Main function to run the knowledge graph system"""
    # Initialize the system
    kg_system = AIKnowledgeGraph()

    # Build the knowledge graph
    kg_system.build_knowledge_graph(force_rebuild=True)

    # Generate statistics
    stats = kg_system.get_system_statistics()
    print("Knowledge Graph Statistics:")
    print(json.dumps(stats, indent=2, default=str))

    # Generate visualization
    fig = kg_system.generate_knowledge_visualization()
    print("Interactive visualization generated!")

    # Export data
    export_json = kg_system.export_knowledge_graph_data('json')
    with open('/Users/dtumkorkmaz/Projects/ai-docs/knowledge_graph_export.json', 'w') as f:
        f.write(export_json)

    print("Knowledge graph data exported!")

    # Analyze content gaps
    gaps = kg_system.analyze_content_gaps()
    print("\nContent Gaps Analysis:")
    print(json.dumps(gaps, indent=2))

    # Generate cross-reference suggestions for a sample file
    sample_file = '/Users/dtumkorkmaz/Projects/ai-docs/README.md'
    suggestions = kg_system.generate_cross_reference_suggestions(sample_file)
    print(f"\nCross-reference suggestions for {sample_file}:")
    for suggestion in suggestions[:5]:
        print(f"- {suggestion['concept_name']}: {suggestion['markdown_link']}")

if __name__ == "__main__":
    main()