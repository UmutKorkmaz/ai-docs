#!/usr/bin/env python3
"""
AI Knowledge Orchestrator - Main Integration System

This is the main orchestrator that integrates all components of the
Advanced Intelligent Cross-Referencing and Knowledge Graph System.

Components Integrated:
- Knowledge Graph Architecture
- Intelligent Cross-Referencing
- Advanced Content Discovery
- Knowledge Visualization
- AI-Powered Insights
- Learning Path Generation
- Content Gap Analysis
"""

import json
import asyncio
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict, Counter
import schedule
import time

# Import our systems
from knowledge_graph_system import AIKnowledgeGraph
from intelligent_cross_referencer import IntelligentCrossReferencer
from content_discovery_system import ContentDiscoverySystem
from knowledge_visualization_system import KnowledgeVisualizationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/dtumkorkmaz/Projects/ai-docs/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """Status of the entire system"""
    knowledge_graph_built: bool
    cross_references_discovered: bool
    topics_discovered: bool
    clusters_created: bool
    visualizations_generated: bool
    last_update: datetime
    total_concepts: int
    total_relationships: int
    total_documents: int
    system_health: float  # 0.0-1.0

@dataclass
class SystemMetrics:
    """System performance and quality metrics"""
    knowledge_graph_density: float
    cross_reference_coverage: float
    topic_coherence_average: float
    content_coverage_score: float
    visualization_count: int
    user_engagement_score: float
    content_freshness_score: float
    knowledge_gap_severity: float

@dataclass
class UserQuery:
    """Represents a user query to the system"""
    query_id: str
    user_id: str
    query_text: str
    query_type: str  # 'search', 'recommendation', 'learning_path', 'exploration'
    context: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    response: Optional[Dict] = None

class AIKnowledgeOrchestrator:
    """Main orchestrator for the AI knowledge system"""

    def __init__(self, base_path: str = "/Users/dtumkorkmaz/Projects/ai-docs"):
        self.base_path = Path(base_path)
        self.startup_time = datetime.now()

        # Initialize all systems
        logger.info("Initializing AI Knowledge Orchestrator...")

        self.knowledge_graph = AIKnowledgeGraph(base_path)
        self.cross_referencer = IntelligentCrossReferencer(base_path)
        self.content_discovery = ContentDiscoverySystem(base_path)
        self.visualization_system = KnowledgeVisualizationSystem(base_path)

        # System status and metrics
        self.system_status = SystemStatus(
            knowledge_graph_built=False,
            cross_references_discovered=False,
            topics_discovered=False,
            clusters_created=False,
            visualizations_generated=False,
            last_update=datetime.now(),
            total_concepts=0,
            total_relationships=0,
            total_documents=0,
            system_health=0.0
        )

        self.system_metrics = SystemMetrics(
            knowledge_graph_density=0.0,
            cross_reference_coverage=0.0,
            topic_coherence_average=0.0,
            content_coverage_score=0.0,
            visualization_count=0,
            user_engagement_score=0.0,
            content_freshness_score=0.0,
            knowledge_gap_severity=0.0
        )

        # User query management
        self.active_queries: Dict[str, UserQuery] = {}
        self.query_history: List[UserQuery] = []

        # Configuration
        self.config = {
            'auto_update_interval': 24,  # hours
            'max_concurrent_queries': 10,
            'cache_duration': 3600,  # seconds
            'enable_auto_cross_refs': True,
            'enable_auto_visualizations': True,
            'enable_learning_paths': True,
            'enable_gap_analysis': True,
            'performance_monitoring': True
        }

        # Load existing state
        self._load_system_state()

        # Start background tasks
        self._start_background_tasks()

    def _load_system_state(self):
        """Load existing system state"""
        state_file = self.base_path / "orchestrator_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)

                    # Restore system status
                    if 'system_status' in state_data:
                        status_data = state_data['system_status']
                        status_data['last_update'] = datetime.fromisoformat(status_data['last_update'])
                        self.system_status = SystemStatus(**status_data)

                    logger.info("System state loaded successfully")
            except Exception as e:
                logger.error(f"Could not load system state: {e}")

    def _save_system_state(self):
        """Save current system state"""
        state_file = self.base_path / "orchestrator_state.json"
        try:
            state_data = {
                'system_status': asdict(self.system_status),
                'system_metrics': asdict(self.system_metrics),
                'last_saved': datetime.now().isoformat(),
                'uptime_hours': (datetime.now() - self.startup_time).total_seconds() / 3600
            }

            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            logger.info("System state saved successfully")
        except Exception as e:
            logger.error(f"Could not save system state: {e}")

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Schedule regular updates
        schedule.every(self.config['auto_update_interval']).hours.do(self._perform_system_update)
        schedule.every(6).hours.do(self._update_system_metrics)
        schedule.every(12).hours.do(self._cleanup_old_data)

        logger.info("Background tasks scheduled")

    def _perform_system_update(self):
        """Perform full system update"""
        logger.info("Starting scheduled system update...")

        try:
            # Update knowledge graph
            self.knowledge_graph.update_knowledge_graph()
            self.system_status.knowledge_graph_built = True

            # Update cross-references
            self.cross_referencer.index_all_files()
            self.system_status.cross_references_discovered = True

            # Update content discovery
            self.content_discovery.discover_topics()
            self.content_discovery.cluster_content()
            self.content_discovery.identify_knowledge_gaps()
            self.system_status.topics_discovered = True
            self.system_status.clusters_created = True

            # Update visualizations
            if self.config['enable_auto_visualizations']:
                self.visualization_system.generate_all_visualizations()
                self.system_status.visualizations_generated = True

            # Update status
            self._update_system_status()
            self._save_system_state()

            logger.info("System update completed successfully")

        except Exception as e:
            logger.error(f"System update failed: {e}")

    def _update_system_metrics(self):
        """Update system performance metrics"""
        logger.info("Updating system metrics...")

        # Knowledge graph density
        if self.knowledge_graph.graph.number_of_nodes() > 0:
            max_edges = self.knowledge_graph.graph.number_of_nodes() ** 2
            actual_edges = self.knowledge_graph.graph.number_of_edges()
            self.system_metrics.knowledge_graph_density = actual_edges / max_edges

        # Cross-reference coverage
        total_docs = len(self.content_discovery.documents)
        if total_docs > 0:
            refs_per_doc = len(self.cross_referencer.cross_references) / total_docs
            self.system_metrics.cross_reference_coverage = min(1.0, refs_per_doc / 10)  # Normalize to 0-1

        # Topic coherence
        if self.content_discovery.topics:
            coherence_scores = [topic.coherence_score for topic in self.content_discovery.topics.values()]
            self.system_metrics.topic_coherence_average = np.mean(coherence_scores)

        # Content coverage
        sections = set(doc['section'] for doc in self.content_discovery.documents.values())
        expected_sections = 25  # Total sections in the documentation
        self.system_metrics.content_coverage_score = len(sections) / expected_sections

        # Knowledge gap severity (inverted - lower is better)
        if self.content_discovery.content_gaps:
            gap_scores = [gap.importance_score for gap in self.content_discovery.content_gaps.values()]
            self.system_metrics.knowledge_gap_severity = 1.0 - np.mean(gap_scores)

        # Calculate overall system health
        self.system_status.system_health = self._calculate_system_health()

        logger.info("System metrics updated")

    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        weights = {
            'knowledge_graph_density': 0.2,
            'cross_reference_coverage': 0.15,
            'topic_coherence_average': 0.15,
            'content_coverage_score': 0.2,
            'knowledge_gap_severity': 0.15,
            'content_freshness_score': 0.15
        }

        health_score = 0.0
        for metric, weight in weights.items():
            value = getattr(self.system_metrics, metric)
            health_score += value * weight

        return min(1.0, health_score)

    def _cleanup_old_data(self):
        """Clean up old cached data"""
        logger.info("Cleaning up old data...")

        # Clean old query history (keep last 1000)
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

        # Clean temporary files older than 7 days
        temp_dir = self.base_path / "temp"
        if temp_dir.exists():
            cutoff_time = datetime.now() - timedelta(days=7)
            for file_path in temp_dir.glob("*"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                    file_path.unlink()

        logger.info("Data cleanup completed")

    def _update_system_status(self):
        """Update system status information"""
        self.system_status.total_concepts = len(self.knowledge_graph.concepts)
        self.system_status.total_relationships = len(self.knowledge_graph.relationships)
        self.system_status.total_documents = len(self.content_discovery.documents)
        self.system_status.last_update = datetime.now()

    def initialize_system(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize the entire system"""
        logger.info("Initializing AI Knowledge System...")
        start_time = datetime.now()

        try:
            # Step 1: Build knowledge graph
            logger.info("Step 1: Building knowledge graph...")
            self.knowledge_graph.build_knowledge_graph(force_rebuild=force_rebuild)
            self.system_status.knowledge_graph_built = True

            # Step 2: Index files for cross-referencing
            logger.info("Step 2: Indexing files for cross-referencing...")
            self.cross_referencer.index_all_files()
            self.system_status.cross_references_discovered = True

            # Step 3: Discover topics and clusters
            logger.info("Step 3: Discovering topics and content clusters...")
            self.content_discovery.discover_topics(force_rebuild=force_rebuild)
            self.content_discovery.cluster_content()
            self.content_discovery.identify_knowledge_gaps()
            self.system_status.topics_discovered = True
            self.system_status.clusters_created = True

            # Step 4: Generate visualizations
            if self.config['enable_auto_visualizations']:
                logger.info("Step 4: Generating visualizations...")
                self.visualization_system.generate_all_visualizations()
                self.system_status.visualizations_generated = True

            # Step 5: Update metrics and status
            self._update_system_status()
            self._update_system_metrics()
            self._save_system_state()

            # Calculate initialization time
            init_time = datetime.now() - start_time

            result = {
                'status': 'success',
                'initialization_time_seconds': init_time.total_seconds(),
                'knowledge_graph': {
                    'concepts': self.system_status.total_concepts,
                    'relationships': self.system_status.total_relationships
                },
                'content_discovery': {
                    'documents': self.system_status.total_documents,
                    'topics': len(self.content_discovery.topics),
                    'clusters': len(self.content_discovery.clusters),
                    'knowledge_gaps': len(self.content_discovery.content_gaps)
                },
                'cross_references': {
                    'total': len(self.cross_referencer.cross_references),
                    'files_indexed': len(self.cross_referencer.file_index)
                },
                'system_health': self.system_status.system_health,
                'visualizations_generated': self.system_status.visualizations_generated
            }

            logger.info(f"System initialization completed in {init_time.total_seconds():.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'initialization_time_seconds': (datetime.now() - start_time).total_seconds()
            }

    async def process_user_query(self, query: UserQuery) -> Dict[str, Any]:
        """Process a user query"""
        logger.info(f"Processing query {query.query_id}: {query.query_text}")

        try:
            # Add to active queries
            self.active_queries[query.query_id] = query

            # Process based on query type
            if query.query_type == 'search':
                response = await self._handle_search_query(query)
            elif query.query_type == 'recommendation':
                response = await self._handle_recommendation_query(query)
            elif query.query_type == 'learning_path':
                response = await self._handle_learning_path_query(query)
            elif query.query_type == 'exploration':
                response = await self._handle_exploration_query(query)
            else:
                response = {'error': f'Unknown query type: {query.query_type}'}

            # Update query
            query.resolved = True
            query.response = response

            # Move to history
            self.active_queries.pop(query.query_id, None)
            self.query_history.append(query)

            return response

        except Exception as e:
            logger.error(f"Error processing query {query.query_id}: {e}")
            return {'error': str(e)}

    async def _handle_search_query(self, query: UserQuery) -> Dict[str, Any]:
        """Handle search queries"""
        search_terms = query.query_text.lower().split()

        # Search in concepts
        concept_matches = []
        for concept_id, concept in self.knowledge_graph.concepts.items():
            if any(term in concept.name.lower() or term in concept.definition.lower()
                   for term in search_terms):
                concept_matches.append({
                    'id': concept_id,
                    'name': concept.name,
                    'section': concept.section,
                    'type': concept.concept_type,
                    'difficulty': concept.difficulty_level,
                    'file_path': concept.file_path
                })

        # Search in documents
        document_matches = []
        for doc_id, doc in self.content_discovery.documents.items():
            content_lower = doc['content'].lower()
            if any(term in content_lower for term in search_terms):
                document_matches.append({
                    'id': doc_id,
                    'title': doc['title'],
                    'section': doc['section'],
                    'category': doc['category'],
                    'file_path': doc['file_path'],
                    'reading_time': doc['reading_time']
                })

        # Get recommendations based on search
        recommendations = []
        if concept_matches:
            top_concept = concept_matches[0]
            recs = self.knowledge_graph.get_concept_recommendations(top_concept['id'], limit=5)
            recommendations = [{
                'name': rec['name'],
                'section': rec['section'],
                'relationship_type': rec['relationship_type'],
                'strength': rec['strength']
            } for rec in recs]

        return {
            'query_type': 'search',
            'search_terms': search_terms,
            'concept_matches': concept_matches,
            'document_matches': document_matches,
            'recommendations': recommendations,
            'total_results': len(concept_matches) + len(document_matches)
        }

    async def _handle_recommendation_query(self, query: UserQuery) -> Dict[str, Any]:
        """Handle recommendation queries"""
        user_profile = query.context.get('user_profile', {})
        content_context = query.context.get('content_context', '')

        recommendations = self.content_discovery.generate_recommendations(
            user_profile=user_profile,
            content_context=content_context
        )

        return {
            'query_type': 'recommendation',
            'recommendations': [
                {
                    'content_id': rec.content_id,
                    'title': rec.title,
                    'section': rec.section,
                    'relevance_score': rec.relevance_score,
                    'reason': rec.reason,
                    'estimated_time': rec.estimated_time,
                    'prerequisites': rec.prerequisites,
                    'learning_objectives': rec.learning_objectives
                }
                for rec in recommendations
            ]
        }

    async def _handle_learning_path_query(self, query: UserQuery) -> Dict[str, Any]:
        """Handle learning path queries"""
        user_goal = query.context.get('goal', query.query_text)
        current_knowledge = query.context.get('current_knowledge', [])
        time_constraint = query.context.get('time_constraint')

        learning_path = self.content_discovery.generate_learning_path(
            user_goal=user_goal,
            current_knowledge=current_knowledge,
            time_constraint=time_constraint
        )

        return {
            'query_type': 'learning_path',
            'learning_path': learning_path
        }

    async def _handle_exploration_query(self, query: UserQuery) -> Dict[str, Any]:
        """Handle exploration queries"""
        concept_name = query.query_text.strip()

        # Find the concept
        concept_id = None
        concept = None
        for cid, c in self.knowledge_graph.concepts.items():
            if concept_name.lower() in c.name.lower():
                concept_id = cid
                concept = c
                break

        if not concept:
            return {'error': f'Concept "{concept_name}" not found'}

        # Get comprehensive information about the concept
        recommendations = self.knowledge_graph.get_concept_recommendations(concept_id, limit=10)
        related_topics = self._get_related_topics(concept_id)
        practical_applications = self._get_practical_applications(concept_id)

        return {
            'query_type': 'exploration',
            'concept': {
                'id': concept_id,
                'name': concept.name,
                'definition': concept.definition,
                'section': concept.section,
                'type': concept.concept_type,
                'difficulty': concept.difficulty_level,
                'tags': concept.tags,
                'file_path': concept.file_path
            },
            'recommendations': recommendations,
            'related_topics': related_topics,
            'practical_applications': practical_applications
        }

    def _get_related_topics(self, concept_id: str) -> List[Dict]:
        """Get topics related to a concept"""
        related_topics = []

        for topic_id, topic in self.content_discovery.topics.items():
            if concept_id in [c['id'] for c in topic.documents if 'id' in c]:
                related_topics.append({
                    'id': topic_id,
                    'name': topic.name,
                    'keywords': topic.keywords,
                    'coherence_score': topic.coherence_score
                })

        return related_topics

    def _get_practical_applications(self, concept_id: str) -> List[Dict]:
        """Get practical applications for a concept"""
        applications = []

        # Find cross-references that are applications
        for ref in self.cross_referencer.cross_references.values():
            if (ref.target_concept.lower() in
                self.knowledge_graph.concepts[concept_id].name.lower() and
                ref.reference_type == 'application'):
                applications.append({
                    'source_file': ref.source_file,
                    'confidence': ref.confidence_score,
                    'context': ref.context_before + "..." + ref.context_after
                })

        return applications

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        logger.info("Generating comprehensive system report...")

        # Update metrics
        self._update_system_metrics()

        # Get statistics from all systems
        kg_stats = self.knowledge_graph.get_system_statistics()
        cross_ref_report = self.cross_referencer.generate_cross_reference_report()
        content_dashboard = self.content_discovery.create_content_discovery_dashboard()

        # Generate insights
        insights = self._generate_system_insights()

        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'system_uptime_hours': (datetime.now() - self.startup_time).total_seconds() / 3600,
                'orchestrator_version': '1.0.0'
            },
            'system_status': asdict(self.system_status),
            'system_metrics': asdict(self.system_metrics),
            'knowledge_graph_stats': kg_stats,
            'cross_reference_analysis': cross_ref_report,
            'content_discovery_dashboard': content_dashboard,
            'system_insights': insights,
            'recommendations': self._generate_system_recommendations(),
            'health_summary': {
                'overall_health': self.system_status.system_health,
                'critical_issues': self._identify_critical_issues(),
                'performance_score': self._calculate_performance_score()
            }
        }

        return report

    def _generate_system_insights(self) -> List[Dict]:
        """Generate intelligent insights about the system"""
        insights = []

        # Knowledge graph insights
        if self.knowledge_graph.graph.number_of_nodes() > 0:
            density = nx.density(self.knowledge_graph.graph)
            if density < 0.1:
                insights.append({
                    'type': 'knowledge_graph',
                    'severity': 'warning',
                    'message': f'Knowledge graph has low density ({density:.3f}). Consider adding more relationships.',
                    'recommendation': 'Run cross-reference discovery to increase connectivity'
                })

        # Content coverage insights
        total_sections = 25  # Expected sections
        covered_sections = len(set(doc['section'] for doc in self.content_discovery.documents.values()))
        coverage_ratio = covered_sections / total_sections

        if coverage_ratio < 0.8:
            insights.append({
                'type': 'content_coverage',
                'severity': 'warning',
                'message': f'Only {coverage_ratio:.1%} of sections are covered.',
                'recommendation': 'Focus on adding content to underrepresented sections'
            })

        # Knowledge gap insights
        high_priority_gaps = [
            gap for gap in self.content_discovery.content_gaps.values()
            if gap.importance_score > 0.7
        ]

        if len(high_priority_gaps) > 10:
            insights.append({
                'type': 'knowledge_gaps',
                'severity': 'critical',
                'message': f'{len(high_priority_gaps)} high-priority knowledge gaps identified.',
                'recommendation': 'Prioritize addressing these gaps to improve content completeness'
            })

        # Topic coherence insights
        if self.content_discovery.topics:
            coherence_scores = [topic.coherence_score for topic in self.content_discovery.topics.values()]
            avg_coherence = np.mean(coherence_scores)

            if avg_coherence < 0.5:
                insights.append({
                    'type': 'topic_coherence',
                    'severity': 'warning',
                    'message': f'Average topic coherence is low ({avg_coherence:.3f}).',
                    'recommendation': 'Review and reorganize content to improve topic clarity'
                })

        # Freshness insights
        old_docs = [
            doc for doc in self.content_discovery.documents.values()
            if doc['last_modified'] < datetime.now() - timedelta(days=30)
        ]

        if len(old_docs) > len(self.content_discovery.documents) * 0.5:
            insights.append({
                'type': 'content_freshness',
                'severity': 'info',
                'message': f'{len(old_docs)} documents haven\'t been updated in 30+ days.',
                'recommendation': 'Consider reviewing and updating older content'
            })

        return insights

    def _generate_system_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations for system improvement"""
        recommendations = []

        # Based on system health
        if self.system_status.system_health < 0.7:
            recommendations.append({
                'priority': 'high',
                'category': 'system_health',
                'action': 'Improve overall system health',
                'description': 'System health is below optimal levels. Focus on improving content coverage and cross-references.',
                'estimated_effort': 'medium'
            })

        # Based on cross-reference coverage
        if self.system_metrics.cross_reference_coverage < 0.5:
            recommendations.append({
                'priority': 'high',
                'category': 'cross_references',
                'action': 'Increase cross-reference coverage',
                'description': 'Add more cross-references to improve content connectivity and discoverability.',
                'estimated_effort': 'low'
            })

        # Based on knowledge gaps
        critical_gaps = [
            gap for gap in self.content_discovery.content_gaps.values()
            if gap.importance_score > 0.8
        ]

        if critical_gaps:
            recommendations.append({
                'priority': 'critical',
                'category': 'content_gaps',
                'action': 'Address critical knowledge gaps',
                'description': f'{len(critical_gaps)} critical knowledge gaps need immediate attention.',
                'estimated_effort': 'high'
            })

        # Based on topic quality
        if self.system_metrics.topic_coherence_average < 0.6:
            recommendations.append({
                'priority': 'medium',
                'category': 'content_quality',
                'action': 'Improve topic coherence',
                'description': 'Review and reorganize content to improve topic clarity and coherence.',
                'estimated_effort': 'medium'
            })

        return recommendations

    def _identify_critical_issues(self) -> List[Dict]:
        """Identify critical issues in the system"""
        issues = []

        # Check for empty or minimal components
        if len(self.knowledge_graph.concepts) < 50:
            issues.append({
                'type': 'knowledge_graph',
                'severity': 'critical',
                'message': 'Very few concepts in knowledge graph',
                'impact': 'Limited knowledge representation and discovery capabilities'
            })

        if len(self.content_discovery.topics) < 5:
            issues.append({
                'type': 'content_discovery',
                'severity': 'critical',
                'message': 'Insufficient topics discovered',
                'impact': 'Poor content organization and topic-based navigation'
            })

        if len(self.cross_referencer.cross_references) < len(self.content_discovery.documents):
            issues.append({
                'type': 'cross_references',
                'severity': 'warning',
                'message': 'Low cross-reference coverage',
                'impact': 'Reduced content connectivity and user guidance'
            })

        return issues

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score"""
        factors = {
            'knowledge_graph_completeness': min(1.0, len(self.knowledge_graph.concepts) / 500),
            'relationship_density': self.system_metrics.knowledge_graph_density,
            'cross_reference_coverage': self.system_metrics.cross_reference_coverage,
            'topic_quality': self.system_metrics.topic_coherence_average,
            'content_coverage': self.system_metrics.content_coverage_score,
            'gap_management': self.system_metrics.knowledge_gap_severity
        }

        weights = {
            'knowledge_graph_completeness': 0.2,
            'relationship_density': 0.15,
            'cross_reference_coverage': 0.15,
            'topic_quality': 0.2,
            'content_coverage': 0.15,
            'gap_management': 0.15
        }

        performance_score = 0.0
        for factor, value in factors.items():
            weight = weights[factor]
            performance_score += value * weight

        return performance_score

    def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        logger.info("Running system diagnostics...")

        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'connectivity': {},
            'performance': {},
            'data_integrity': {},
            'overall_status': 'healthy'
        }

        # Test knowledge graph
        try:
            kg_stats = self.knowledge_graph.get_system_statistics()
            diagnostics['components']['knowledge_graph'] = {
                'status': 'healthy',
                'concepts': kg_stats['concepts']['total'],
                'relationships': kg_stats['relationships']['total'],
                'graph_density': kg_stats['graph_metrics']['density']
            }
        except Exception as e:
            diagnostics['components']['knowledge_graph'] = {
                'status': 'error',
                'error': str(e)
            }
            diagnostics['overall_status'] = 'degraded'

        # Test cross-referencer
        try:
            validation = self.cross_referencer.validate_cross_references()
            diagnostics['components']['cross_referencer'] = {
                'status': 'healthy',
                'total_cross_refs': len(self.cross_referencer.cross_references),
                'broken_links': len(validation['broken_links']),
                'validation_issues': len(validation['broken_links']) + len(validation['missing_targets'])
            }

            if len(validation['broken_links']) > 0:
                diagnostics['overall_status'] = 'degraded'
        except Exception as e:
            diagnostics['components']['cross_referencer'] = {
                'status': 'error',
                'error': str(e)
            }
            diagnostics['overall_status'] = 'degraded'

        # Test content discovery
        try:
            topics = self.content_discovery.discover_topics()
            diagnostics['components']['content_discovery'] = {
                'status': 'healthy',
                'documents': len(self.content_discovery.documents),
                'topics_discovered': len(topics),
                'knowledge_gaps': len(self.content_discovery.content_gaps)
            }
        except Exception as e:
            diagnostics['components']['content_discovery'] = {
                'status': 'error',
                'error': str(e)
            }
            diagnostics['overall_status'] = 'degraded'

        # Test visualization system
        try:
            # Test if visualization can be created
            test_fig = self.visualization_system.create_interactive_knowledge_graph()
            diagnostics['components']['visualization'] = {
                'status': 'healthy',
                'test_visualization_created': True
            }
        except Exception as e:
            diagnostics['components']['visualization'] = {
                'status': 'error',
                'error': str(e)
            }
            diagnostics['overall_status'] = 'degraded'

        # Performance metrics
        diagnostics['performance'] = {
            'system_health': self.system_status.system_health,
            'uptime_hours': (datetime.now() - self.startup_time).total_seconds() / 3600,
            'memory_usage': self._estimate_memory_usage(),
            'last_update': self.system_status.last_update.isoformat()
        }

        # Data integrity checks
        diagnostics['data_integrity'] = {
            'knowledge_graph_consistent': self._check_knowledge_graph_consistency(),
            'cross_references_valid': len(self.cross_referencer.validate_cross_references()['broken_links']) == 0,
            'content_index_complete': len(self.content_discovery.documents) > 0
        }

        return diagnostics

    def _estimate_memory_usage(self) -> str:
        """Estimate memory usage of the system"""
        total_objects = (
            len(self.knowledge_graph.concepts) +
            len(self.knowledge_graph.relationships) +
            len(self.content_discovery.documents) +
            len(self.cross_referencer.cross_references)
        )

        # Rough estimation
        estimated_mb = total_objects * 0.001  # Very rough estimate

        if estimated_mb < 100:
            return f"{estimated_mb:.1f} MB"
        else:
            return f"{estimated_mb / 1000:.1f} GB"

    def _check_knowledge_graph_consistency(self) -> bool:
        """Check knowledge graph consistency"""
        try:
            # Check for orphaned relationships
            concept_ids = set(self.knowledge_graph.concepts.keys())
            for rel in self.knowledge_graph.relationships:
                if rel.source_id not in concept_ids or rel.target_id not in concept_ids:
                    return False

            # Check graph consistency
            return self.knowledge_graph.graph.number_of_nodes() == len(self.knowledge_graph.concepts)
        except:
            return False

    def start_background_scheduler(self):
        """Start the background scheduler"""
        logger.info("Starting background scheduler...")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down AI Knowledge Orchestrator...")

        # Save final state
        self._save_system_state()

        # Generate final report
        final_report = self.generate_comprehensive_report()

        with open(self.base_path / "final_system_report.json", 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        logger.info("System shutdown complete")

def main():
    """Main function to run the orchestrator"""
    import sys

    orchestrator = AIKnowledgeOrchestrator()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "init":
            # Initialize the system
            result = orchestrator.initialize_system(force_rebuild=True)
            print("Initialization Result:")
            print(json.dumps(result, indent=2))

        elif command == "diagnostics":
            # Run diagnostics
            diagnostics = orchestrator.run_system_diagnostics()
            print("System Diagnostics:")
            print(json.dumps(diagnostics, indent=2))

        elif command == "report":
            # Generate comprehensive report
            report = orchestrator.generate_comprehensive_report()
            print("System Report:")
            print(json.dumps(report, indent=2, default=str))

        elif command == "scheduler":
            # Start background scheduler
            print("Starting background scheduler... Press Ctrl+C to stop")
            try:
                orchestrator.start_background_scheduler()
            except KeyboardInterrupt:
                print("\nScheduler stopped")
                orchestrator.shutdown()

        else:
            print(f"Unknown command: {command}")
            print("Available commands: init, diagnostics, report, scheduler")
    else:
        # Default: initialize and run diagnostics
        print("AI Knowledge Orchestrator")
        print("=" * 50)

        # Initialize
        init_result = orchestrator.initialize_system()
        print(f"Initialization: {init_result['status']}")

        # Run diagnostics
        diagnostics = orchestrator.run_system_diagnostics()
        print(f"System Status: {diagnostics['overall_status']}")
        print(f"System Health: {diagnostics['performance']['system_health']:.2f}")

        print("\nUse 'python ai_knowledge_orchestrator.py --help' for available commands")

if __name__ == "__main__":
    main()