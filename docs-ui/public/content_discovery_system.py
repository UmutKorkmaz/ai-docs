#!/usr/bin/env python3
"""
Advanced Content Discovery and Topic Modeling System for AI Documentation

This system provides:
- Topic modeling and content clustering
- Concept-based navigation and exploration
- Learning path generation based on knowledge gaps
- Intelligent content recommendations
- Content gap identification and recommendations
- Emerging concept detection and trending analysis
- Expert system for answering complex AI questions
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
import hashlib

# ML and NLP libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
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

# Import our systems
from knowledge_graph_system import AIKnowledgeGraph
from intelligent_cross_referencer import IntelligentCrossReferencer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Topic:
    """Represents a discovered topic"""
    id: str
    name: str
    description: str
    keywords: List[str]
    documents: List[str]
    coherence_score: float
    topic_distribution: Dict[str, float]  # document_id -> probability
    created_at: datetime = datetime.now()

@dataclass
class ContentCluster:
    """Represents a cluster of related content"""
    id: str
    name: str
    centroid: np.ndarray
    documents: List[str]
    cluster_score: float
    dominant_topics: List[str]
    learning_objectives: List[str]
    difficulty_level: float

@dataclass
class ContentRecommendation:
    """Represents a content recommendation"""
    content_id: str
    title: str
    file_path: str
    section: str
    relevance_score: float
    reason: str
    estimated_time: int  # in minutes
    prerequisites: List[str]
    learning_objectives: List[str]

@dataclass
class KnowledgeGap:
    """Represents a knowledge gap"""
    gap_type: str  # 'missing_concept', 'prerequisite', 'application', 'advanced_topic'
    description: str
    importance_score: float
    suggested_content: List[str]
    current_mastery_level: float
    target_mastery_level: float

class ContentDiscoverySystem:
    """Main content discovery system"""

    def __init__(self, base_path: str = "/Users/dtumkorkmaz/Projects/ai-docs"):
        self.base_path = Path(base_path)
        self.knowledge_graph = AIKnowledgeGraph(base_path)
        self.cross_referencer = IntelligentCrossReferencer(base_path)

        # Initialize NLP components
        self._initialize_nlp()

        # Configuration
        self.config = {
            'min_topic_coherence': 0.3,
            'max_topics': 20,
            'min_cluster_size': 3,
            'max_recommendations': 10,
            'embedding_dim': 100,
            'min_gap_importance': 0.5,
            'trending_threshold': 0.1,  # Minimum growth rate for trending detection
            'content_update_frequency': 7  # days
        }

        # Data structures
        self.documents: Dict[str, Dict] = {}  # document_id -> content metadata
        self.topics: Dict[str, Topic] = {}
        self.clusters: Dict[str, ContentCluster] = {}
        self.content_embeddings: Dict[str, np.ndarray] = {}
        self.user_profiles: Dict[str, Dict] = {}  # User learning profiles
        self.trending_topics: Dict[str, float] = {}
        self.content_gaps: Dict[str, KnowledgeGap] = {}

        # Load existing data
        self._load_existing_data()

        # Build content index
        self._build_content_index()

    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            # Download NLTK data if needed
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            logger.info("NLP components initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize NLP components: {e}")
            self.stop_words = set()
            self.lemmatizer = None

    def _load_existing_data(self):
        """Load existing discovery data"""
        cache_file = self.base_path / "content_discovery_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.topics = {
                        topic_id: Topic(**topic_data)
                        for topic_id, topic_data in data.get('topics', {}).items()
                    }
                    self.trending_topics = data.get('trending_topics', {})
                    self.content_gaps = {
                        gap_id: KnowledgeGap(**gap_data)
                        for gap_id, gap_data in data.get('content_gaps', {}).items()
                    }
                logger.info(f"Loaded {len(self.topics)} topics and {len(self.content_gaps)} gaps")
            except Exception as e:
                logger.warning(f"Could not load discovery cache: {e}")

    def _save_cache(self):
        """Save current discovery data"""
        cache_file = self.base_path / "content_discovery_cache.json"
        try:
            data = {
                'topics': {
                    topic_id: asdict(topic) for topic_id, topic in self.topics.items()
                },
                'trending_topics': self.trending_topics,
                'content_gaps': {
                    gap_id: asdict(gap) for gap_id, gap in self.content_gaps.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Content discovery cache saved")
        except Exception as e:
            logger.error(f"Could not save cache: {e}")

    def _build_content_index(self):
        """Build content index from markdown files"""
        logger.info("Building content index...")

        md_files = list(self.base_path.glob("**/*.md"))
        logger.info(f"Processing {len(md_files)} markdown files...")

        for file_path in md_files:
            self._process_document(file_path)

        logger.info(f"Indexed {len(self.documents)} documents")

    def _process_document(self, file_path: Path):
        """Process a single document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract metadata
            doc_id = str(file_path)
            sections = self._parse_sections(content)
            word_count = len(content.split())
            reading_time = max(1, word_count // 200)  # 200 words per minute

            # Extract key terms
            key_terms = self._extract_key_terms(content)

            # Determine section and category
            section = self._determine_section(file_path)
            category = self._determine_category(content, sections)

            self.documents[doc_id] = {
                'file_path': str(file_path),
                'title': self._extract_title(content),
                'content': content,
                'sections': sections,
                'word_count': word_count,
                'reading_time': reading_time,
                'key_terms': key_terms,
                'section': section,
                'category': category,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                'size': len(content)
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    def _parse_sections(self, content: str) -> List[Dict]:
        """Parse markdown sections"""
        sections = []
        lines = content.split('\n')
        current_section = None

        for i, line in enumerate(lines):
            if line.startswith('#'):
                if current_section:
                    sections.append(current_section)

                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                current_section = {
                    'title': title,
                    'level': level,
                    'line_start': i,
                    'content': []
                }
            elif current_section and line.strip():
                current_section['content'].append(line)

        if current_section:
            sections.append(current_section)

        return sections

    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())

        # Filter out common words
        filtered_words = [
            word for word in words
            if word not in self.stop_words and len(word) > 3
        ]

        # Count frequencies
        word_freq = Counter(filtered_words)

        # Get top terms
        top_terms = [word for word, freq in word_freq.most_common(20)]

        return top_terms

    def _extract_title(self, content: str) -> str:
        """Extract title from content"""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return "Untitled"

    def _determine_section(self, file_path: Path) -> str:
        """Determine section from file path"""
        path_str = str(file_path)

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

    def _determine_category(self, content: str, sections: List[Dict]) -> str:
        """Determine content category"""
        content_lower = content.lower()

        category_keywords = {
            'Theory': ['theory', 'mathematical', 'algorithm', 'proof', 'concept'],
            'Implementation': ['code', 'python', 'implement', 'example', 'practical'],
            'Tutorial': ['tutorial', 'step-by-step', 'how to', 'guide', 'learn'],
            'Research': ['research', 'paper', 'study', 'experiment', 'analysis'],
            'Application': ['application', 'use case', 'industry', 'real-world', 'practical'],
            'Reference': ['reference', 'documentation', 'api', 'specification', 'manual']
        }

        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(content_lower.count(keyword) for keyword in keywords)
            category_scores[category] = score

        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]

        return 'General'

    def discover_topics(self, force_rebuild: bool = False) -> Dict[str, Topic]:
        """Discover topics using Latent Dirichlet Allocation"""
        if self.topics and not force_rebuild:
            logger.info("Topics already discovered, returning cached results")
            return self.topics

        logger.info("Discovering topics...")

        # Prepare documents
        documents = []
        doc_ids = []

        for doc_id, doc_data in self.documents.items():
            # Preprocess content
            processed_content = self._preprocess_text(doc_data['content'])
            if len(processed_content.split()) > 50:  # Only include substantial documents
                documents.append(processed_content)
                doc_ids.append(doc_id)

        if len(documents) < 5:
            logger.warning("Not enough documents for topic modeling")
            return {}

        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # Determine optimal number of topics
        n_topics = min(self.config['max_topics'], len(documents) // 3)
        n_topics = max(5, n_topics)  # At least 5 topics

        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100,
            learning_method='online'
        )

        lda.fit(tfidf_matrix)

        # Extract topics
        topics = {}
        for topic_idx in range(n_topics):
            # Get top words for this topic
            top_words_idx = lda.components_[topic_idx].argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]

            # Get topic distribution across documents
            topic_dist = lda.transform(tfidf_matrix)[:, topic_idx]

            # Calculate coherence score
            coherence = self._calculate_topic_coherence(top_words, documents)

            # Create topic object
            topic_id = f"topic_{topic_idx}"
            topic = Topic(
                id=topic_id,
                name=self._generate_topic_name(top_words),
                description=f"Topic focusing on: {', '.join(top_words[:5])}",
                keywords=top_words,
                documents=[doc_ids[i] for i, prob in enumerate(topic_dist) if prob > 0.1],
                coherence_score=coherence,
                topic_distribution={
                    doc_ids[i]: float(prob) for i, prob in enumerate(topic_dist) if prob > 0.05
                }
            )

            if coherence >= self.config['min_topic_coherence']:
                topics[topic_id] = topic

        self.topics = topics
        self._save_cache()

        logger.info(f"Discovered {len(topics)} coherent topics")
        return topics

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for topic modeling"""
        # Convert to lowercase
        text = text.lower()

        # Remove markdown syntax
        text = re.sub(r'[#*`\[\]()_~]', ' ', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)

        # Tokenize and lemmatize
        if self.lemmatizer:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            text = ' '.join(tokens)

        return text

    def _calculate_topic_coherence(self, words: List[str], documents: List[str]) -> float:
        """Calculate topic coherence score"""
        # Simple coherence measure based on word co-occurrence
        coherence = 0.0
        word_pairs = 0

        for i, word1 in enumerate(words[:5]):  # Use top 5 words
            for word2 in words[i+1:5]:
                cooccurrence = 0
                for doc in documents:
                    if word1 in doc and word2 in doc:
                        cooccurrence += 1

                if cooccurrence > 0:
                    coherence += cooccurrence
                    word_pairs += 1

        return coherence / word_pairs if word_pairs > 0 else 0.0

    def _generate_topic_name(self, words: List[str]) -> str:
        """Generate a human-readable topic name"""
        # Filter out common AI terms to get more specific names
        common_terms = {'learning', 'model', 'data', 'algorithm', 'method', 'approach'}
        specific_words = [w for w in words[:5] if w not in common_terms]

        if specific_words:
            return ' '.join(specific_words[:3]).title()
        else:
            return ' '.join(words[:3]).title()

    def cluster_content(self, n_clusters: int = 10) -> Dict[str, ContentCluster]:
        """Cluster content based on similarity"""
        logger.info("Clustering content...")

        # Generate document embeddings
        if not self.content_embeddings:
            self._generate_document_embeddings()

        if len(self.content_embeddings) < n_clusters:
            n_clusters = max(2, len(self.content_embeddings) // 2)

        # Prepare data
        doc_ids = list(self.content_embeddings.keys())
        embeddings = np.array([self.content_embeddings[doc_id] for doc_id in doc_ids])

        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Create clusters
        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_docs = [doc_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]

            if len(cluster_docs) >= self.config['min_cluster_size']:
                # Calculate cluster characteristics
                centroid = kmeans.cluster_centers_[cluster_id]
                dominant_topics = self._get_dominant_topics(cluster_docs)
                difficulty = self._calculate_cluster_difficulty(cluster_docs)
                objectives = self._generate_cluster_objectives(cluster_docs)

                cluster = ContentCluster(
                    id=f"cluster_{cluster_id}",
                    name=self._generate_cluster_name(dominant_topics, cluster_docs),
                    centroid=centroid,
                    documents=cluster_docs,
                    cluster_score=self._calculate_cluster_score(cluster_docs),
                    dominant_topics=dominant_topics,
                    learning_objectives=objectives,
                    difficulty_level=difficulty
                )
                clusters[cluster.id] = cluster

        self.clusters = clusters
        logger.info(f"Created {len(clusters)} content clusters")
        return clusters

    def _generate_document_embeddings(self):
        """Generate embeddings for all documents"""
        logger.info("Generating document embeddings...")

        # Use TF-IDF for embeddings
        documents = [doc['content'] for doc in self.documents.values()]
        doc_ids = list(self.documents.keys())

        vectorizer = TfidfVectorizer(
            max_features=self.config['embedding_dim'],
            stop_words='english',
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(documents)

        for i, doc_id in enumerate(doc_ids):
            self.content_embeddings[doc_id] = tfidf_matrix[i].toarray().flatten()

        logger.info(f"Generated embeddings for {len(doc_ids)} documents")

    def _get_dominant_topics(self, doc_ids: List[str]) -> List[str]:
        """Get dominant topics for a set of documents"""
        topic_counts = Counter()

        for doc_id in doc_ids:
            for topic_id, topic in self.topics.items():
                if doc_id in topic.documents:
                    topic_counts[topic_id] += 1

        # Get top topics
        dominant_topics = [topic_id for topic_id, count in topic_counts.most_common(3)]
        return dominant_topics

    def _calculate_cluster_difficulty(self, doc_ids: List[str]) -> float:
        """Calculate average difficulty of documents in cluster"""
        difficulties = []

        for doc_id in doc_ids:
            if doc_id in self.documents:
                # Estimate difficulty based on content
                doc = self.documents[doc_id]
                difficulty = self._estimate_document_difficulty(doc)
                difficulties.append(difficulty)

        return np.mean(difficulties) if difficulties else 2.5

    def _estimate_document_difficulty(self, doc: Dict) -> float:
        """Estimate difficulty level of a document"""
        difficulty = 2.5  # Default medium difficulty

        content = doc['content'].lower()

        # Check for difficulty indicators
        advanced_terms = [
            'mathematical', 'theoretical', 'complex', 'advanced',
            'optimization', 'calculus', 'linear algebra', 'probability theory',
            'deep learning', 'neural architecture', 'research'
        ]

        basic_terms = [
            'introduction', 'basic', 'beginner', 'simple', 'overview',
            'getting started', 'tutorial', 'fundamentals'
        ]

        advanced_score = sum(content.count(term) for term in advanced_terms)
        basic_score = sum(content.count(term) for term in basic_terms)

        if advanced_score > basic_score * 2:
            difficulty = 4.0
        elif basic_score > advanced_score * 2:
            difficulty = 1.5

        # Adjust based on section
        section_difficulty = {
            'Foundational ML': 2.0,
            'Advanced Deep Learning': 4.0,
            'Future Directions': 5.0,
            'Industry Applications': 3.0,
            'AI Ethics': 3.5
        }

        section_diff = section_difficulty.get(doc['section'], 2.5)
        difficulty = (difficulty + section_diff) / 2

        return min(5.0, max(1.0, difficulty))

    def _generate_cluster_objectives(self, doc_ids: List[str]) -> List[str]:
        """Generate learning objectives for a cluster"""
        objectives = []

        # Analyze common themes in documents
        all_sections = []
        all_terms = []

        for doc_id in doc_ids:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                all_sections.extend([section['title'] for section in doc['sections']])
                all_terms.extend(doc['key_terms'])

        # Generate objectives based on common themes
        section_counter = Counter(all_sections)
        term_counter = Counter(all_terms)

        if section_counter:
            top_section = section_counter.most_common(1)[0][0]
            objectives.append(f"Understand {top_section.lower()}")

        if term_counter:
            top_terms = [term for term, count in term_counter.most_common(3)]
            objectives.append(f"Learn about {', '.join(top_terms)}")

        # Add general objectives
        objectives.extend([
            "Apply concepts in practical scenarios",
            "Connect theory with real-world applications"
        ])

        return objectives[:4]  # Limit to 4 objectives

    def _calculate_cluster_score(self, doc_ids: List[str]) -> float:
        """Calculate cluster quality score"""
        if len(doc_ids) < 2:
            return 0.0

        # Calculate average similarity within cluster
        similarities = []

        for i, doc1_id in enumerate(doc_ids):
            for doc2_id in doc_ids[i+1:]:
                if doc1_id in self.content_embeddings and doc2_id in self.content_embeddings:
                    emb1 = self.content_embeddings[doc1_id]
                    emb2 = self.content_embeddings[doc2_id]
                    similarity = cosine_similarity([emb1], [emb2])[0][0]
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _generate_cluster_name(self, dominant_topics: List[str], doc_ids: List[str]) -> str:
        """Generate a name for the cluster"""
        if dominant_topics:
            topic_names = [self.topics[topic_id].name for topic_id in dominant_topics if topic_id in self.topics]
            if topic_names:
                return ' & '.join(topic_names[:2])

        # Fallback to using document terms
        all_terms = []
        for doc_id in doc_ids:
            if doc_id in self.documents:
                all_terms.extend(self.documents[doc_id]['key_terms'][:5])

        if all_terms:
            term_counter = Counter(all_terms)
            top_terms = [term for term, count in term_counter.most_common(3)]
            return ' & '.join(top_terms)

        return "General Topic Cluster"

    def generate_recommendations(self, user_profile: Dict = None,
                              content_context: str = None,
                              max_recommendations: int = None) -> List[ContentRecommendation]:
        """Generate personalized content recommendations"""
        if max_recommendations is None:
            max_recommendations = self.config['max_recommendations']

        recommendations = []

        # Generate recommendations based on different strategies
        if user_profile:
            # User-based recommendations
            user_recs = self._generate_user_based_recommendations(user_profile)
            recommendations.extend(user_recs)

        if content_context:
            # Content-based recommendations
            content_recs = self._generate_content_based_recommendations(content_context)
            recommendations.extend(content_recs)

        # Knowledge gap-based recommendations
        gap_recs = self._generate_gap_based_recommendations()
        recommendations.extend(gap_recs)

        # Remove duplicates and sort by relevance
        unique_recs = self._deduplicate_recommendations(recommendations)
        unique_recs.sort(key=lambda x: x.relevance_score, reverse=True)

        return unique_recs[:max_recommendations]

    def _generate_user_based_recommendations(self, user_profile: Dict) -> List[ContentRecommendation]:
        """Generate recommendations based on user profile"""
        recommendations = []

        # Get user's interests and current level
        interests = user_profile.get('interests', [])
        current_level = user_profile.get('level', 'intermediate')
        completed_docs = user_profile.get('completed_documents', [])

        # Find similar users (collaborative filtering)
        similar_users = self._find_similar_users(user_profile)

        # Recommend content liked by similar users
        for similar_user in similar_users:
            for doc_id in similar_user.get('liked_documents', []):
                if doc_id not in completed_docs and doc_id in self.documents:
                    doc = self.documents[doc_id]

                    # Calculate relevance based on interests
                    relevance = self._calculate_content_relevance(doc, interests, current_level)

                    if relevance > 0.3:
                        rec = ContentRecommendation(
                            content_id=doc_id,
                            title=doc['title'],
                            file_path=doc['file_path'],
                            section=doc['section'],
                            relevance_score=relevance,
                            reason=f"Users with similar interests found this helpful",
                            estimated_time=doc['reading_time'],
                            prerequisites=self._get_prerequisites(doc_id),
                            learning_objectives=self._get_learning_objectives(doc_id)
                        )
                        recommendations.append(rec)

        return recommendations

    def _generate_content_based_recommendations(self, content_context: str) -> List[ContentRecommendation]:
        """Generate recommendations based on content context"""
        recommendations = []

        # Find documents similar to the context
        similar_docs = self._find_similar_documents(content_context)

        for doc_id, similarity in similar_docs:
            if doc_id in self.documents:
                doc = self.documents[doc_id]

                rec = ContentRecommendation(
                    content_id=doc_id,
                    title=doc['title'],
                    file_path=doc['file_path'],
                    section=doc['section'],
                    relevance_score=similarity,
                    reason="Similar to current content",
                    estimated_time=doc['reading_time'],
                    prerequisites=self._get_prerequisites(doc_id),
                    learning_objectives=self._get_learning_objectives(doc_id)
                )
                recommendations.append(rec)

        return recommendations

    def _generate_gap_based_recommendations(self) -> List[ContentRecommendation]:
        """Generate recommendations based on knowledge gaps"""
        recommendations = []

        for gap_id, gap in self.content_gaps.items():
            if gap.importance_score >= self.config['min_gap_importance']:
                for content_id in gap.suggested_content:
                    if content_id in self.documents:
                        doc = self.documents[content_id]

                        rec = ContentRecommendation(
                            content_id=content_id,
                            title=doc['title'],
                            file_path=doc['file_path'],
                            section=doc['section'],
                            relevance_score=gap.importance_score,
                            reason=f"Addresses knowledge gap: {gap.description}",
                            estimated_time=doc['reading_time'],
                            prerequisites=self._get_prerequisites(content_id),
                            learning_objectives=gap.description
                        )
                        recommendations.append(rec)

        return recommendations

    def _find_similar_users(self, user_profile: Dict) -> List[Dict]:
        """Find users with similar profiles (mock implementation)"""
        # In a real system, this would query a user database
        # For now, return mock similar users
        return [
            {
                'id': 'user_1',
                'liked_documents': list(self.documents.keys())[:5],
                'interests': user_profile.get('interests', [])
            }
        ]

    def _find_similar_documents(self, content_context: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find documents similar to content context"""
        # Generate embedding for context
        context_embedding = self._get_text_embedding(content_context)

        if context_embedding is None:
            return []

        similarities = []
        for doc_id, doc_embedding in self.content_embeddings.items():
            similarity = cosine_similarity([context_embedding], [doc_embedding])[0][0]
            similarities.append((doc_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text"""
        # Use TF-IDF vectorizer
        if hasattr(self, 'vectorizer'):
            try:
                embedding = self.vectorizer.transform([text]).toarray().flatten()
                return embedding
            except:
                pass

        # Fallback to simple word count
        words = text.lower().split()
        embedding = np.zeros(100)  # Fixed size
        for i, word in enumerate(words[:100]):
            embedding[i] = hash(word) % 1000 / 1000  # Simple hash-based embedding

        return embedding

    def _calculate_content_relevance(self, doc: Dict, interests: List[str],
                                   user_level: str) -> float:
        """Calculate relevance score for content"""
        relevance = 0.0

        # Interest match
        doc_text = (doc['title'] + ' ' + ' '.join(doc['key_terms'])).lower()
        interest_matches = sum(1 for interest in interests if interest.lower() in doc_text)
        relevance += interest_matches * 0.3

        # Level match
        doc_difficulty = self._estimate_document_difficulty(doc)
        level_scores = {'beginner': 1.5, 'intermediate': 2.5, 'advanced': 3.5, 'expert': 4.5}
        target_level = level_scores.get(user_level, 2.5)

        level_match = 1.0 - abs(doc_difficulty - target_level) / 4.0
        relevance += level_match * 0.2

        # Content quality (based on word count and structure)
        quality_score = min(1.0, doc['word_count'] / 1000)  # Normalize to 0-1
        relevance += quality_score * 0.1

        return min(1.0, relevance)

    def _get_prerequisites(self, doc_id: str) -> List[str]:
        """Get prerequisites for a document"""
        # Use knowledge graph to find prerequisites
        doc = self.documents.get(doc_id)
        if not doc:
            return []

        prerequisites = []
        for concept_id, concept in self.knowledge_graph.concepts.items():
            if concept.file_path == doc_id:
                # Find prerequisite concepts
                for rel in self.knowledge_graph.relationships:
                    if rel.target_id == concept_id and rel.relationship_type == 'prerequisite':
                        if rel.source_id in self.knowledge_graph.concepts:
                            prereq_concept = self.knowledge_graph.concepts[rel.source_id]
                            prerequisites.append(prereq_concept.name)

        return prerequisites[:5]  # Limit to 5 prerequisites

    def _get_learning_objectives(self, doc_id: str) -> List[str]:
        """Get learning objectives for a document"""
        doc = self.documents.get(doc_id)
        if not doc:
            return []

        objectives = []

        # Extract objectives from section titles
        for section in doc['sections']:
            if any(keyword in section['title'].lower()
                   for keyword in ['learn', 'understand', 'master', 'implement']):
                objectives.append(section['title'])

        # Generate generic objectives based on content
        if doc['category'] == 'Theory':
            objectives.append("Understand theoretical foundations")
        elif doc['category'] == 'Implementation':
            objectives.append("Implement practical solutions")
        elif doc['category'] == 'Tutorial':
            objectives.append("Follow step-by-step guidance")

        return objectives[:3]  # Limit to 3 objectives

    def _deduplicate_recommendations(self, recommendations: List[ContentRecommendation]) -> List[ContentRecommendation]:
        """Remove duplicate recommendations"""
        seen_ids = set()
        unique_recs = []

        for rec in recommendations:
            if rec.content_id not in seen_ids:
                seen_ids.add(rec.content_id)
                unique_recs.append(rec)

        return unique_recs

    def identify_knowledge_gaps(self) -> Dict[str, KnowledgeGap]:
        """Identify knowledge gaps in the documentation"""
        logger.info("Identifying knowledge gaps...")

        gaps = {}

        # Check for missing foundational concepts
        foundation_gaps = self._identify_foundation_gaps()
        gaps.update(foundation_gaps)

        # Check for prerequisite gaps
        prereq_gaps = self._identify_prerequisite_gaps()
        gaps.update(prereq_gaps)

        # Check for application gaps
        application_gaps = self._identify_application_gaps()
        gaps.update(application_gaps)

        # Check for advanced topic gaps
        advanced_gaps = self._identify_advanced_topic_gaps()
        gaps.update(advanced_gaps)

        self.content_gaps = gaps
        self._save_cache()

        logger.info(f"Identified {len(gaps)} knowledge gaps")
        return gaps

    def _identify_foundation_gaps(self) -> Dict[str, KnowledgeGap]:
        """Identify gaps in foundational concepts"""
        gaps = {}

        # Define expected foundational concepts
        expected_foundations = [
            'machine learning basics',
            'linear algebra',
            'probability theory',
            'statistics',
            'calculus',
            'python programming',
            'data preprocessing',
            'model evaluation'
        ]

        # Check coverage
        for concept in expected_foundations:
            coverage = self._check_concept_coverage(concept)

            if coverage < 0.7:  # Less than 70% coverage
                gap_id = f"foundation_{hash(concept) % 10000}"
                gaps[gap_id] = KnowledgeGap(
                    gap_type='missing_concept',
                    description=f"Limited coverage of {concept}",
                    importance_score=0.8,
                    suggested_content=self._suggest_foundation_content(concept),
                    current_mastery_level=coverage,
                    target_mastery_level=0.9
                )

        return gaps

    def _identify_prerequisite_gaps(self) -> Dict[str, KnowledgeGap]:
        """Identify prerequisite chain gaps"""
        gaps = {}

        # Check knowledge graph for missing prerequisite relationships
        for concept_id, concept in self.knowledge_graph.concepts.items():
            missing_prereqs = []

            # Find concepts that should be prerequisites but aren't documented
            expected_prereqs = self._get_expected_prerequisites(concept)

            for expected_prereq in expected_prereqs:
                if not self._has_documented_prerequisite(concept_id, expected_prereq):
                    missing_prereqs.append(expected_prereq)

            if missing_prereqs:
                gap_id = f"prereq_{concept_id}"
                gaps[gap_id] = KnowledgeGap(
                    gap_type='prerequisite',
                    description=f"Missing prerequisites for {concept.name}: {', '.join(missing_prereqs)}",
                    importance_score=0.7,
                    suggested_content=self._suggest_prerequisite_content(missing_prereqs),
                    current_mastery_level=0.5,
                    target_mastery_level=0.8
                )

        return gaps

    def _identify_application_gaps(self) -> Dict[str, KnowledgeGap]:
        """Identify gaps in practical applications"""
        gaps = {}

        # Check for theoretical concepts without practical applications
        theory_concepts = [
            concept for concept in self.knowledge_graph.concepts.values()
            if concept.concept_type in ['methods', 'techniques']
        ]

        for concept in theory_concepts:
            has_applications = self._has_practical_applications(concept.id)

            if not has_applications:
                gap_id = f"application_{concept.id}"
                gaps[gap_id] = KnowledgeGap(
                    gap_type='application',
                    description=f"Lack of practical applications for {concept.name}",
                    importance_score=0.6,
                    suggested_content=self._suggest_application_content(concept),
                    current_mastery_level=0.7,
                    target_mastery_level=0.9
                )

        return gaps

    def _identify_advanced_topic_gaps(self) -> Dict[str, KnowledgeGap]:
        """Identify gaps in advanced topics"""
        gaps = {}

        # Define expected advanced topics for 2024-2025
        expected_advanced = [
            'multimodal ai',
            'generative ai ethics',
            'ai safety research',
            'quantum machine learning',
            'neuromorphic computing',
            'federated learning',
            'explainable ai',
            'ai governance',
            'climate ai',
            'cultural ai'
        ]

        for topic in expected_advanced:
            coverage = self._check_concept_coverage(topic)

            if coverage < 0.5:  # Less than 50% coverage
                gap_id = f"advanced_{hash(topic) % 10000}"
                gaps[gap_id] = KnowledgeGap(
                    gap_type='advanced_topic',
                    description=f"Limited coverage of advanced topic: {topic}",
                    importance_score=0.7,
                    suggested_content=self._suggest_advanced_content(topic),
                    current_mastery_level=coverage,
                    target_mastery_level=0.8
                )

        return gaps

    def _check_concept_coverage(self, concept: str) -> float:
        """Check how well a concept is covered in the documentation"""
        concept_lower = concept.lower()
        total_docs = len(self.documents)
        covering_docs = 0

        for doc_id, doc in self.documents.items():
            content = doc['content'].lower()
            if concept_lower in content:
                covering_docs += 1

        return covering_docs / total_docs if total_docs > 0 else 0.0

    def _get_expected_prerequisites(self, concept: AIConcept) -> List[str]:
        """Get expected prerequisites for a concept"""
        # Define prerequisite mappings
        prereq_mapping = {
            'neural networks': ['linear algebra', 'calculus', 'machine learning basics'],
            'deep learning': ['neural networks', 'python programming', 'data preprocessing'],
            'transformer': ['attention mechanism', 'sequence modeling', 'neural networks'],
            'gan': ['neural networks', 'generative models', 'optimization'],
            'reinforcement learning': ['machine learning basics', 'probability theory', 'optimization']
        }

        concept_name = concept.name.lower()
        for key, prereqs in prereq_mapping.items():
            if key in concept_name:
                return prereqs

        return []

    def _has_documented_prerequisite(self, concept_id: str, prereq_name: str) -> bool:
        """Check if a prerequisite is documented"""
        prereq_lower = prereq_name.lower()

        # Check if prerequisite exists as a concept
        for concept in self.knowledge_graph.concepts.values():
            if prereq_lower in concept.name.lower():
                # Check if there's a documented relationship
                for rel in self.knowledge_graph.relationships:
                    if (rel.source_id == concept.id and rel.target_id == concept_id and
                        rel.relationship_type == 'prerequisite'):
                        return True

        return False

    def _has_practical_applications(self, concept_id: str) -> bool:
        """Check if a concept has practical applications"""
        for rel in self.knowledge_graph.relationships:
            if (rel.source_id == concept_id and
                rel.relationship_type in ['application', 'example']):
                return True

        return False

    def _suggest_foundation_content(self, concept: str) -> List[str]:
        """Suggest content for foundational concepts"""
        # Find documents that partially cover the concept
        suggestions = []

        for doc_id, doc in self.documents.items():
            if concept.lower() in doc['content'].lower():
                suggestions.append(doc_id)

        return suggestions[:3]

    def _suggest_prerequisite_content(self, missing_prereqs: List[str]) -> List[str]:
        """Suggest content for missing prerequisites"""
        suggestions = []

        for prereq in missing_prereqs:
            prereq_lower = prereq.lower()

            for doc_id, doc in self.documents.items():
                if prereq_lower in doc['content'].lower() and doc_id not in suggestions:
                    suggestions.append(doc_id)
                    break

        return suggestions

    def _suggest_application_content(self, concept: AIConcept) -> List[str]:
        """Suggest application content for a concept"""
        # Find practical implementation documents
        suggestions = []

        for doc_id, doc in self.documents.items():
            if (doc['category'] in ['Implementation', 'Tutorial'] and
                concept.name.lower() in doc['content'].lower()):
                suggestions.append(doc_id)

        return suggestions

    def _suggest_advanced_content(self, topic: str) -> List[str]:
        """Suggest advanced content for a topic"""
        suggestions = []

        for doc_id, doc in self.documents.items():
            if (doc['section'] in ['Future Directions', 'Emerging Research', 'Advanced Deep Learning'] and
                topic.lower() in doc['content'].lower()):
                suggestions.append(doc_id)

        return suggestions

    def detect_trending_topics(self) -> Dict[str, float]:
        """Detect trending topics based on content updates and references"""
        logger.info("Detecting trending topics...")
")

        trending = {}

        # Analyze recent content updates
        recent_docs = [
            doc for doc in self.documents.values()
            if doc['last_modified'] > datetime.now() - timedelta(days=30)
        ]

        # Count mentions of topics in recent documents
        topic_mentions = Counter()

        for doc in recent_docs:
            for topic_id, topic in self.topics.items():
                for keyword in topic.keywords:
                    if keyword in doc['content'].lower():
                        topic_mentions[topic_id] += 1

        # Calculate trending scores
        total_mentions = sum(topic_mentions.values())
        if total_mentions > 0:
            for topic_id, mentions in topic_mentions.items():
                trending_score = mentions / total_mentions

                if trending_score >= self.config['trending_threshold']:
                    trending[topic_id] = trending_score

        self.trending_topics = trending
        logger.info(f"Detected {len(trending)} trending topics")
        return trending

    def generate_learning_path(self, user_goal: str, current_knowledge: List[str] = None,
                             time_constraint: int = None) -> Dict:
        """Generate personalized learning path"""
        logger.info(f"Generating learning path for goal: {user_goal}")

        if current_knowledge is None:
            current_knowledge = []

        # Identify target concepts based on goal
        target_concepts = self._identify_goal_concepts(user_goal)

        # Find optimal path through knowledge graph
        path_concepts = self._find_learning_path(current_knowledge, target_concepts)

        # Calculate estimated time
        total_time = sum(
            self.documents.get(concept, {}).get('reading_time', 30)
            for concept in path_concepts
            if concept in self.documents
        )

        # Filter by time constraint if provided
        if time_constraint and total_time > time_constraint:
            path_concepts = self._filter_by_time_constraint(path_concepts, time_constraint)
            total_time = sum(
                self.documents.get(concept, {}).get('reading_time', 30)
                for concept in path_concepts
                if concept in self.documents
            )

        # Generate milestones
        milestones = self._generate_learning_milestones(path_concepts)

        learning_path = {
            'goal': user_goal,
            'current_knowledge': current_knowledge,
            'target_concepts': target_concepts,
            'learning_path': path_concepts,
            'estimated_time_minutes': total_time,
            'estimated_time_hours': total_time / 60,
            'milestones': milestones,
            'prerequisites_to_cover': self._identify_missing_prerequisites(current_knowledge, path_concepts),
            'difficulty_progression': self._calculate_difficulty_progression(path_concepts)
        }

        return learning_path

    def _identify_goal_concepts(self, user_goal: str) -> List[str]:
        """Identify target concepts based on user goal"""
        goal_lower = user_goal.lower()
        target_concepts = []

        # Map goals to concepts
        goal_mapping = {
            'learn machine learning': ['machine learning basics', 'supervised learning', 'model evaluation'],
            'master deep learning': ['neural networks', 'deep learning', 'backpropagation', 'optimization'],
            'become nlp expert': ['natural language processing', 'transformers', 'language models', 'text processing'],
            'computer vision specialist': ['computer vision', 'image processing', 'object detection', 'image classification'],
            'ai ethics researcher': ['ai ethics', 'bias detection', 'fairness', 'interpretability', 'safety'],
            'generative ai developer': ['generative ai', 'gan', 'vae', 'diffusion models', 'creativity']
        }

        for goal_pattern, concepts in goal_mapping.items():
            if goal_pattern in goal_lower:
                target_concepts.extend(concepts)
                break

        # If no specific mapping, find related concepts
        if not target_concepts:
            for concept_id, concept in self.knowledge_graph.concepts.items():
                if any(word in goal_lower for word in concept.name.lower().split()):
                    target_concepts.append(concept_id)

        return target_concepts[:10]  # Limit to 10 target concepts

    def _find_learning_path(self, current_knowledge: List[str],
                          target_concepts: List[str]) -> List[str]:
        """Find optimal learning path through knowledge graph"""
        if not target_concepts:
            return []

        # Use knowledge graph to find paths
        all_paths = []

        for target in target_concepts:
            for current in current_knowledge:
                path = self.knowledge_graph.discover_learning_path(current, target)
                if path:
                    all_paths.append(path)

        # If no paths found, return target concepts directly
        if not all_paths:
            return target_concepts

        # Combine and optimize paths
        combined_path = []
        seen_concepts = set(current_knowledge)

        # Sort paths by length and add concepts
        all_paths.sort(key=len)

        for path in all_paths:
            for concept in path:
                if concept not in seen_concepts:
                    combined_path.append(concept)
                    seen_concepts.add(concept)

        return combined_path

    def _filter_by_time_constraint(self, concepts: List[str], time_limit: int) -> List[str]:
        """Filter concepts by time constraint"""
        filtered_concepts = []
        current_time = 0

        for concept in concepts:
            if concept in self.documents:
                concept_time = self.documents[concept].get('reading_time', 30)

                if current_time + concept_time <= time_limit:
                    filtered_concepts.append(concept)
                    current_time += concept_time
            else:
                filtered_concepts.append(concept)

        return filtered_concepts

    def _generate_learning_milestones(self, concepts: List[str]) -> List[Dict]:
        """Generate learning milestones"""
        milestones = []
        milestone_size = max(3, len(concepts) // 4)  # Create 3-4 milestones

        for i in range(0, len(concepts), milestone_size):
            milestone_concepts = concepts[i:i + milestone_size]

            # Determine milestone theme
            milestone_theme = self._determine_milestone_theme(milestone_concepts)

            milestone = {
                'title': f"Milestone {i // milestone_size + 1}: {milestone_theme}",
                'concepts': milestone_concepts,
                'estimated_time': sum(
                    self.documents.get(concept, {}).get('reading_time', 30)
                    for concept in milestone_concepts
                    if concept in self.documents
                ),
                'objectives': self._generate_milestone_objectives(milestone_concepts)
            }

            milestones.append(milestone)

        return milestones

    def _determine_milestone_theme(self, concepts: List[str]) -> str:
        """Determine theme for a group of concepts"""
        # Count sections represented by concepts
        section_counts = Counter()

        for concept in concepts:
            if concept in self.documents:
                section = self.documents[concept]['section']
                section_counts[section] += 1

        if section_counts:
            dominant_section = section_counts.most_common(1)[0][0]
            return f"Master {dominant_section}"

        return "Core Concepts"

    def _generate_milestone_objectives(self, concepts: List[str]) -> List[str]:
        """Generate objectives for a milestone"""
        objectives = []

        # Extract key terms from concepts
        all_terms = []
        for concept in concepts:
            if concept in self.documents:
                all_terms.extend(self.documents[concept]['key_terms'][:3])

        # Generate objectives based on common themes
        if all_terms:
            term_counter = Counter(all_terms)
            top_terms = [term for term, count in term_counter.most_common(3)]

            for term in top_terms:
                objectives.append(f"Understand {term}")

        objectives.append("Complete practical exercises")
        objectives.append("Connect concepts to real applications")

        return objectives[:4]

    def _identify_missing_prerequisites(self, current_knowledge: List[str],
                                      path_concepts: List[str]) -> List[str]:
        """Identify missing prerequisites for the learning path"""
        missing_prereqs = []

        for concept in path_concepts:
            if concept in self.knowledge_graph.concepts:
                concept_obj = self.knowledge_graph.concepts[concept]

                # Check if prerequisites are met
                for rel in self.knowledge_graph.relationships:
                    if (rel.target_id == concept and
                        rel.relationship_type == 'prerequisite' and
                        rel.source_id in self.knowledge_graph.concepts):

                        prereq_concept = self.knowledge_graph.concepts[rel.source_id]
                        if prereq_concept.name not in current_knowledge:
                            missing_prereqs.append(prereq_concept.name)

        return list(set(missing_prereqs))

    def _calculate_difficulty_progression(self, concepts: List[str]) -> List[float]:
        """Calculate difficulty progression through learning path"""
        difficulties = []

        for concept in concepts:
            if concept in self.documents:
                difficulty = self._estimate_document_difficulty(self.documents[concept])
                difficulties.append(difficulty)
            else:
                difficulties.append(2.5)  # Default difficulty

        return difficulties

    def create_content_discovery_dashboard(self) -> Dict:
        """Create comprehensive dashboard for content discovery"""
        logger.info("Creating content discovery dashboard...")

        # Ensure all analyses are up to date
        self.discover_topics()
        self.cluster_content()
        self.identify_knowledge_gaps()
        self.detect_trending_topics()

        dashboard = {
            'overview': {
                'total_documents': len(self.documents),
                'total_topics': len(self.topics),
                'total_clusters': len(self.clusters),
                'knowledge_gaps': len(self.content_gaps),
                'trending_topics': len(self.trending_topics)
            },
            'content_analysis': {
                'documents_by_section': dict(Counter(
                    doc['section'] for doc in self.documents.values()
                )),
                'documents_by_category': dict(Counter(
                    doc['category'] for doc in self.documents.values()
                )),
                'average_reading_time': np.mean([
                    doc['reading_time'] for doc in self.documents.values()
                ]),
                'difficulty_distribution': self._get_difficulty_distribution()
            },
            'topic_analysis': {
                'topics_by_coherence': sorted(
                    [(topic.name, topic.coherence_score) for topic in self.topics.values()],
                    key=lambda x: x[1], reverse=True
                )[:10],
                'topic_distribution': {
                    topic.name: len(topic.documents)
                    for topic in self.topics.values()
                }
            },
            'cluster_analysis': {
                'clusters_by_size': sorted(
                    [(cluster.name, len(cluster.documents)) for cluster in self.clusters.values()],
                    key=lambda x: x[1], reverse=True
                ),
                'cluster_difficulty_levels': {
                    cluster.name: cluster.difficulty_level
                    for cluster in self.clusters.values()
                }
            },
            'knowledge_gaps': {
                'gaps_by_importance': sorted(
                    [(gap.description, gap.importance_score) for gap in self.content_gaps.values()],
                    key=lambda x: x[1], reverse=True
                )[:10],
                'gap_types': dict(Counter(
                    gap.gap_type for gap in self.content_gaps.values()
                ))
            },
            'trending_analysis': {
                'trending_topics': [
                    (self.topics[topic_id].name, score)
                    for topic_id, score in sorted(
                        self.trending_topics.items(),
                        key=lambda x: x[1], reverse=True
                    )
                ] if self.trending_topics else [],
                'recent_activity': self._get_recent_activity()
            },
            'recommendations': self._generate_system_recommendations()
        }

        return dashboard

    def _get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of content difficulty levels"""
        difficulties = [
            self._estimate_document_difficulty(doc)
            for doc in self.documents.values()
        ]

        difficulty_bins = {
            'Beginner (1-2)': 0,
            'Intermediate (2-3)': 0,
            'Advanced (3-4)': 0,
            'Expert (4-5)': 0
        }

        for difficulty in difficulties:
            if difficulty <= 2:
                difficulty_bins['Beginner (1-2)'] += 1
            elif difficulty <= 3:
                difficulty_bins['Intermediate (2-3)'] += 1
            elif difficulty <= 4:
                difficulty_bins['Advanced (3-4)'] += 1
            else:
                difficulty_bins['Expert (4-5)'] += 1

        return difficulty_bins

    def _get_recent_activity(self) -> List[Dict]:
        """Get recent content activity"""
        recent_docs = [
            doc for doc in self.documents.values()
            if doc['last_modified'] > datetime.now() - timedelta(days=7)
        ]

        return [
            {
                'title': doc['title'],
                'file_path': doc['file_path'],
                'section': doc['section'],
                'last_modified': doc['last_modified'].isoformat()
            }
            for doc in recent_docs[:10]
        ]

    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []

        # Content coverage recommendations
        if len(self.content_gaps) > 10:
            recommendations.append(
                f"Address {len(self.content_gaps)} identified knowledge gaps to improve coverage"
            )

        # Topic coherence recommendations
        low_coherence_topics = [
            topic for topic in self.topics.values()
            if topic.coherence_score < 0.5
        ]

        if len(low_coherence_topics) > len(self.topics) * 0.3:
            recommendations.append(
                "Consider improving topic coherence by reorganizing related content"
            )

        # Update frequency recommendations
        old_docs = [
            doc for doc in self.documents.values()
            if doc['last_modified'] < datetime.now() - timedelta(days=30)
        ]

        if len(old_docs) > len(self.documents) * 0.5:
            recommendations.append(
                f"Consider updating {len(old_docs)} documents that haven't been modified in 30+ days"
            )

        # Cross-reference recommendations
        total_refs = len(self.cross_referencer.cross_references)
        if total_refs < len(self.documents) * 2:
            recommendations.append(
                "Add more cross-references to improve content connectivity"
            )

        return recommendations

def main():
    """Main function to run the content discovery system"""
    # Initialize the system
    discovery_system = ContentDiscoverySystem()

    # Discover topics
    topics = discovery_system.discover_topics(force_rebuild=True)
    print(f"Discovered {len(topics)} topics")

    # Cluster content
    clusters = discovery_system.cluster_content()
    print(f"Created {len(clusters)} content clusters")

    # Identify knowledge gaps
    gaps = discovery_system.identify_knowledge_gaps()
    print(f"Identified {len(gaps)} knowledge gaps")

    # Detect trending topics
    trending = discovery_system.detect_trending_topics()
    print(f"Detected {len(trending)} trending topics")

    # Generate sample recommendations
    recommendations = discovery_system.generate_recommendations(
        user_profile={
            'interests': ['machine learning', 'deep learning'],
            'level': 'intermediate'
        }
    )
    print(f"Generated {len(recommendations)} recommendations")

    # Generate sample learning path
    learning_path = discovery_system.generate_learning_path(
        user_goal="master deep learning",
        current_knowledge=["python programming", "basic mathematics"],
        time_constraint=300  # 5 hours
    )
    print(f"Generated learning path with {len(learning_path['learning_path'])} concepts")

    # Create dashboard
    dashboard = discovery_system.create_content_discovery_dashboard()

    # Save results
    results = {
        'dashboard': dashboard,
        'learning_path': learning_path,
        'recommendations': [asdict(rec) for rec in recommendations[:5]]
    }

    with open('/Users/dtumkorkmaz/Projects/ai-docs/content_discovery_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("Content discovery analysis complete! Results saved to content_discovery_results.json")

if __name__ == "__main__":
    main()