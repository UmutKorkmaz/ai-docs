#!/usr/bin/env python3
"""
Advanced Knowledge Visualization System for AI Documentation

This system provides:
- Interactive knowledge graph visualization
- Concept relationship diagrams
- Learning progress tracking
- Content heat maps and analytics
- 3D concept space visualization
- Interactive dashboard for exploration
- Real-time content analytics
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime, timedelta
import hashlib
from collections import defaultdict, Counter

# Visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import FancyBboxPatch
    import networkx as nx
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import plotly.offline as pyo
except ImportError as e:
    print(f"Warning: Some visualization dependencies not available: {e}")

# Import our systems
from knowledge_graph_system import AIKnowledgeGraph
from intelligent_cross_referencer import IntelligentCrossReferencer
from content_discovery_system import ContentDiscoverySystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    width: int = 1200
    height: int = 800
    color_scheme: str = 'plotly'
    layout: str = 'spring'  # spring, circular, hierarchical
    node_size_range: Tuple[int, int] = (10, 50)
    edge_width_range: Tuple[float, float] = (1, 10)
    show_labels: bool = True
    label_threshold: float = 0.1
    interactive: bool = True
    save_format: str = 'html'  # html, png, svg

@dataclass
class AnalyticsData:
    """Analytics data for visualization"""
    total_concepts: int
    total_relationships: int
    total_documents: int
    topic_distribution: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    section_coverage: Dict[str, float]
    cross_reference_density: float
    learning_progress: Dict[str, float]
    trending_topics: List[Tuple[str, float]]
    knowledge_gaps: List[Dict]

class KnowledgeVisualizationSystem:
    """Main knowledge visualization system"""

    def __init__(self, base_path: str = "/Users/dtumkorkmaz/Projects/ai-docs"):
        self.base_path = Path(base_path)
        self.knowledge_graph = AIKnowledgeGraph(base_path)
        self.cross_referencer = IntelligentCrossReferencer(base_path)
        self.discovery_system = ContentDiscoverySystem(base_path)

        # Visualization configuration
        self.config = VisualizationConfig()

        # Color schemes
        self.color_schemes = {
            'plotly': px.colors.qualitative.Plotly,
            'set3': px.colors.qualitative.Set3,
            'pastel': px.colors.qualitative.Pastel,
            'dark': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            'section': {
                'Foundational ML': '#FF6B6B',
                'Advanced Deep Learning': '#4ECDC4',
                'NLP': '#45B7D1',
                'Computer Vision': '#96CEB4',
                'Generative AI': '#FFEAA7',
                'AI Agents': '#DDA0DD',
                'AI Ethics': '#F4A460',
                'Industry Applications': '#98D8C8',
                'Interdisciplinary': '#F7DC6F',
                'Technical': '#BB8FCE'
            }
        }

        # Layout algorithms
        self.layout_algorithms = {
            'spring': self._spring_layout,
            'circular': self._circular_layout,
            'hierarchical': self._hierarchical_layout,
            'force_directed': self._force_directed_layout,
            'spectral': self._spectral_layout
        }

        # Initialize analytics data
        self.analytics_data = None

    def create_interactive_knowledge_graph(self, config: VisualizationConfig = None) -> go.Figure:
        """Create interactive knowledge graph visualization"""
        if config:
            self.config = config

        logger.info("Creating interactive knowledge graph visualization...")

        # Ensure knowledge graph is built
        if not self.knowledge_graph.graph.nodes():
            self.knowledge_graph.build_knowledge_graph()

        # Generate layout
        pos = self._generate_layout()

        # Create node traces
        node_traces = self._create_node_traces(pos)

        # Create edge traces
        edge_traces = self._create_edge_traces(pos)

        # Combine traces
        fig = go.Figure(data=edge_traces + node_traces)

        # Update layout
        fig.update_layout(
            title={
                'text': "AI Knowledge Graph - Interactive Visualization",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=self.config.width,
            height=self.config.height,
            annotations=[
                dict(
                    text="Hover over nodes to see details | Drag to rearrange | Double-click to reset",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.005,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=10)
                )
            ]
        )

        # Add interactivity
        fig.update_traces(
            selector=dict(type='scatter'),
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Section: %{customdata[1]}<br>' +
                         'Type: %{customdata[2]}<br>' +
                         'Difficulty: %{customdata[3]}' +
                         '<extra></extra>'
        )

        return fig

    def _generate_layout(self) -> Dict:
        """Generate node positions using specified layout algorithm"""
        layout_func = self.layout_algorithms.get(
            self.config.layout, self._spring_layout
        )
        return layout_func()

    def _spring_layout(self) -> Dict:
        """Generate spring layout"""
        return nx.spring_layout(
            self.knowledge_graph.graph,
            k=1,
            iterations=50,
            seed=42
        )

    def _circular_layout(self) -> Dict:
        """Generate circular layout"""
        return nx.circular_layout(self.knowledge_graph.graph)

    def _hierarchical_layout(self) -> Dict:
        """Generate hierarchical layout based on difficulty"""
        # Group nodes by difficulty level
        difficulty_groups = defaultdict(list)
        for node, data in self.knowledge_graph.graph.nodes(data=True):
            difficulty = data.get('difficulty', 2)
            difficulty_groups[difficulty].append(node)

        # Position nodes in layers
        pos = {}
        layer_spacing = 2.0
        node_spacing = 1.0

        for difficulty, nodes in sorted(difficulty_groups.items()):
            layer_y = difficulty * layer_spacing
            n_nodes = len(nodes)

            for i, node in enumerate(nodes):
                x = (i - n_nodes/2) * node_spacing
                pos[node] = (x, layer_y)

        return pos

    def _force_directed_layout(self) -> Dict:
        """Generate force-directed layout"""
        return nx.kamada_kawai_layout(self.knowledge_graph.graph)

    def _spectral_layout(self) -> Dict:
        """Generate spectral layout"""
        return nx.spectral_layout(self.knowledge_graph.graph)

    def _create_node_traces(self, pos: Dict) -> List[go.Scatter]:
        """Create node traces for visualization"""
        # Extract node data
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        node_hover_data = []

        for node in self.knowledge_graph.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            node_data = self.knowledge_graph.graph.nodes[node]

            # Node information
            name = node_data.get('name', node)
            section = node_data.get('section', 'Unknown')
            concept_type = node_data.get('type', 'Unknown')
            difficulty = node_data.get('difficulty', 2)

            node_text.append(name if self.config.show_labels else '')
            node_hover_data.append([name, section, concept_type, difficulty])

            # Color by section
            color = self.color_schemes['section'].get(section, '#95a5a6')
            node_color.append(color)

            # Size by importance (degree + difficulty)
            degree = self.knowledge_graph.graph.degree(node)
            size = self.config.node_size_range[0] + (degree * 2 + difficulty * 3)
            node_size.append(size)

        # Create main node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            customdata=node_hover_data,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            textfont=dict(size=10, color='black'),
            textposition='middle center'
        )

        return [node_trace]

    def _create_edge_traces(self, pos: Dict) -> List[go.Scatter]:
        """Create edge traces for visualization"""
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in self.knowledge_graph.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Edge information
            edge_data = edge[2]
            relationship_type = edge_data.get('relationship_type', 'related')
            strength = edge_data.get('strength', 0.5)
            edge_info.append(f"{relationship_type}: {strength:.2f}")

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#95a5a6'),
            hoverinfo='none',
            mode='lines'
        )

        return [edge_trace]

    def create_topic_visualization(self) -> go.Figure:
        """Create topic model visualization"""
        logger.info("Creating topic visualization...")

        # Ensure topics are discovered
        if not self.discovery_system.topics:
            self.discovery_system.discover_topics()

        if not self.discovery_system.topics:
            logger.warning("No topics found for visualization")
            return go.Figure()

        # Prepare data
        topics_data = []
        for topic_id, topic in self.discovery_system.topics.items():
            topics_data.append({
                'topic_id': topic_id,
                'name': topic.name,
                'coherence': topic.coherence_score,
                'num_documents': len(topic.documents),
                'keywords': ', '.join(topic.keywords[:5])
            })

        df = pd.DataFrame(topics_data)

        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Topic Coherence Scores',
                'Document Distribution per Topic',
                'Topic Keywords',
                'Topic Network'
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "table"}, {"type": "scatter"}]
            ]
        )

        # Topic coherence bar chart
        fig.add_trace(
            go.Bar(
                x=df['name'],
                y=df['coherence'],
                marker_color='lightblue',
                name='Coherence Score'
            ),
            row=1, col=1
        )

        # Document distribution pie chart
        fig.add_trace(
            go.Pie(
                labels=df['name'],
                values=df['num_documents'],
                name='Document Count'
            ),
            row=1, col=2
        )

        # Keywords table
        fig.add_trace(
            go.Table(
                header=dict(values=['Topic', 'Top Keywords']),
                cells=dict(values=[df['name'], df['keywords']])
            ),
            row=2, col=1
        )

        # Topic network (simplified)
        topic_pos = nx.spring_layout(nx.complete_graph(len(df)))
        topic_x = [topic_pos[i][0] for i in range(len(df))]
        topic_y = [topic_pos[i][1] for i in range(len(df))]

        fig.add_trace(
            go.Scatter(
                x=topic_x,
                y=topic_y,
                mode='markers+text',
                text=df['name'],
                marker=dict(
                    size=df['coherence'] * 50,
                    color=df['num_documents'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Documents", x=1.1)
                ),
                name='Topics'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title="Topic Model Visualization",
            height=800,
            showlegend=False
        )

        return fig

    def create_learning_progress_dashboard(self, user_data: Dict = None) -> go.Figure:
        """Create learning progress dashboard"""
        logger.info("Creating learning progress dashboard...")

        # Generate mock user data if not provided
        if user_data is None:
            user_data = self._generate_mock_user_data()

        # Create subplot dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Overall Progress',
                'Topic Mastery',
                'Study Time Trend',
                'Difficulty Progression',
                'Section Completion',
                'Recent Activity',
                'Knowledge Gaps',
                'Achievement Badges',
                'Recommendations'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "table"}, {"type": "table"}]
            ]
        )

        # Overall progress indicator
        overall_progress = user_data.get('overall_progress', 65)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_progress,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Completion %"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )

        # Topic mastery bar chart
        topics = user_data.get('topic_mastery', {})
        fig.add_trace(
            go.Bar(
                x=list(topics.keys()),
                y=list(topics.values()),
                marker_color='lightgreen',
                name='Mastery Level'
            ),
            row=1, col=2
        )

        # Study time trend
        study_data = user_data.get('study_time_history', [])
        if study_data:
            dates = [entry['date'] for entry in study_data]
            times = [entry['minutes'] for entry in study_data]
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=times,
                    mode='lines+markers',
                    name='Study Time (min)'
                ),
                row=1, col=3
            )

        # Difficulty progression
        difficulty_progress = user_data.get('difficulty_progression', {})
        fig.add_trace(
            go.Bar(
                x=list(difficulty_progress.keys()),
                y=list(difficulty_progress.values()),
                marker_color='orange',
                name='Difficulty Level'
            ),
            row=2, col=1
        )

        # Section completion pie chart
        sections = user_data.get('section_completion', {})
        fig.add_trace(
            go.Pie(
                labels=list(sections.keys()),
                values=list(sections.values()),
                name="Completion by Section"
            ),
            row=2, col=2
        )

        # Recent activity
        recent_activity = user_data.get('recent_activity', [])
        if recent_activity:
            activity_dates = [activity['date'] for activity in recent_activity]
            activity_counts = [activity['activities'] for activity in recent_activity]
            fig.add_trace(
                go.Bar(
                    x=activity_dates,
                    y=activity_counts,
                    marker_color='purple',
                    name='Activities'
                ),
                row=2, col=3
            )

        # Knowledge gaps
        gaps = user_data.get('knowledge_gaps', [])
        gap_names = [gap['name'] for gap in gaps[:5]]
        gap_scores = [gap['urgency'] for gap in gaps[:5]]
        fig.add_trace(
            go.Bar(
                x=gap_names,
                y=gap_scores,
                marker_color='red',
                name='Knowledge Gaps'
            ),
            row=3, col=1
        )

        # Achievement badges table
        badges = user_data.get('achievements', [])
        badge_data = [[badge['name'], badge['earned_date']] for badge in badges]
        fig.add_trace(
            go.Table(
                header=dict(values=['Achievement', 'Earned Date']),
                cells=dict(values=list(zip(*badge_data)) if badge_data else [[], []])
            ),
            row=3, col=2
        )

        # Recommendations table
        recommendations = user_data.get('recommendations', [])
        rec_data = [[rec['title'], rec['priority']] for rec in recommendations[:5]]
        fig.add_trace(
            go.Table(
                header=dict(values=['Recommendation', 'Priority']),
                cells=dict(values=list(zip(*rec_data)) if rec_data else [[], []])
            ),
            row=3, col=3
        )

        fig.update_layout(
            title="Learning Progress Dashboard",
            height=1200,
            showlegend=False
        )

        return fig

    def _generate_mock_user_data(self) -> Dict:
        """Generate mock user data for demonstration"""
        return {
            'overall_progress': 65,
            'topic_mastery': {
                'Machine Learning': 80,
                'Deep Learning': 60,
                'NLP': 45,
                'Computer Vision': 30,
                'AI Ethics': 70
            },
            'study_time_history': [
                {'date': '2024-01-01', 'minutes': 30},
                {'date': '2024-01-02', 'minutes': 45},
                {'date': '2024-01-03', 'minutes': 60},
                {'date': '2024-01-04', 'minutes': 25},
                {'date': '2024-01-05', 'minutes': 90}
            ],
            'difficulty_progression': {
                'Beginner': 100,
                'Intermediate': 70,
                'Advanced': 30,
                'Expert': 10
            },
            'section_completion': {
                'Foundational ML': 85,
                'Advanced Deep Learning': 60,
                'NLP': 40,
                'Computer Vision': 25,
                'AI Ethics': 75
            },
            'recent_activity': [
                {'date': '2024-01-05', 'activities': 5},
                {'date': '2024-01-04', 'activities': 3},
                {'date': '2024-01-03', 'activities': 7}
            ],
            'knowledge_gaps': [
                {'name': 'Advanced Mathematics', 'urgency': 8},
                {'name': 'Research Methods', 'urgency': 6},
                {'name': 'Industry Applications', 'urgency': 5}
            ],
            'achievements': [
                {'name': 'First Steps', 'earned_date': '2024-01-01'},
                {'name': 'Week Warrior', 'earned_date': '2024-01-07'},
                {'name': 'Knowledge Explorer', 'earned_date': '2024-01-15'}
            ],
            'recommendations': [
                {'title': 'Complete Neural Networks Basics', 'priority': 'High'},
                {'title': 'Practice with Real Datasets', 'priority': 'Medium'},
                {'title': 'Join Study Group', 'priority': 'Low'}
            ]
        }

    def create_content_heatmap(self) -> go.Figure:
        """Create content coverage heatmap"""
        logger.info("Creating content heatmap...")

        # Get content data
        sections = list(set(doc['section'] for doc in self.discovery_system.documents.values()))
        topics = list(self.discovery_system.topics.keys())

        # Create coverage matrix
        coverage_matrix = np.zeros((len(sections), len(topics)))

        for i, section in enumerate(sections):
            for j, topic_id in enumerate(topics):
                topic = self.discovery_system.topics[topic_id]
                section_docs = [
                    doc_id for doc_id, doc in self.discovery_system.documents.items()
                    if doc['section'] == section
                ]
                topic_docs_in_section = len(set(topic.documents) & set(section_docs))
                total_section_docs = len(section_docs)

                if total_section_docs > 0:
                    coverage_matrix[i, j] = topic_docs_in_section / total_section_docs

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=coverage_matrix,
            x=[self.discovery_system.topics[tid].name for tid in topics],
            y=sections,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Coverage Ratio")
        ))

        fig.update_layout(
            title="Content Coverage Heatmap by Section and Topic",
            xaxis_title="Topics",
            yaxis_title="Sections",
            width=1000,
            height=600
        )

        return fig

    def create_3d_concept_space(self) -> go.Figure:
        """Create 3D visualization of concept space"""
        logger.info("Creating 3D concept space visualization...")

        # Generate document embeddings if not available
        if not self.discovery_system.content_embeddings:
            self.discovery_system._generate_document_embeddings()

        if not self.discovery_system.content_embeddings:
            logger.warning("No embeddings available for 3D visualization")
            return go.Figure()

        # Convert embeddings to numpy array
        embeddings = np.array(list(self.discovery_system.content_embeddings.values()))
        doc_ids = list(self.discovery_system.content_embeddings.keys())

        # Reduce to 3 dimensions using PCA
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)

        # Create color mapping by section
        sections = [
            self.discovery_system.documents[doc_id]['section']
            for doc_id in doc_ids
        ]
        unique_sections = list(set(sections))
        colors = [self.color_schemes['section'].get(section, '#95a5a6') for section in sections]

        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.6
            ),
            text=[
                self.discovery_system.documents[doc_id]['title']
                for doc_id in doc_ids
            ],
            hovertemplate='<b>%{text}</b><br>' +
                         'Section: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=sections
        )])

        fig.update_layout(
            title="3D Concept Space Visualization",
            scene=dict(
                xaxis_title="PCA Component 1",
                yaxis_title="PCA Component 2",
                zaxis_title="PCA Component 3"
            ),
            width=900,
            height=700
        )

        return fig

    def create_cross_reference_network(self) -> go.Figure:
        """Create cross-reference network visualization"""
        logger.info("Creating cross-reference network visualization...")

        # Get cross-reference data
        cross_refs = self.cross_referencer.cross_references

        if not cross_refs:
            logger.warning("No cross-references found for visualization")
            return go.Figure()

        # Build network graph from cross-references
        G = nx.DiGraph()

        for ref_id, ref in cross_refs.items():
            G.add_node(ref.source_file, type='source')
            G.add_node(ref.target_file, type='target')
            G.add_edge(
                ref.source_file,
                ref.target_file,
                weight=ref.confidence_score,
                type=ref.reference_type
            )

        # Generate layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_info = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(Path(node).stem)
            node_info.append(f"File: {node}")

            # Color by degree
            degree = G.degree(node)
            node_color.append(degree)

        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            weight = edge[2].get('weight', 0.5)
            ref_type = edge[2].get('type', 'unknown')
            edge_info.append(f"{ref_type}: {weight:.2f}")

        # Create figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))

        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            customdata=node_info,
            marker=dict(
                size=node_color,
                color=node_color,
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title="Connections"),
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=8),
            textposition='middle center'
        ))

        fig.update_layout(
            title="Cross-Reference Network Visualization",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        return fig

    def create_analytics_dashboard(self) -> go.Figure:
        """Create comprehensive analytics dashboard"""
        logger.info("Creating analytics dashboard...")

        # Generate analytics data
        self._generate_analytics_data()

        # Create subplot dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Document Distribution',
                'Topic Distribution',
                'Difficulty Levels',
                'Section Coverage',
                'Cross-Reference Density',
                'Trending Topics',
                'Knowledge Gaps',
                'Content Growth',
                'Learning Path Completion'
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]
            ]
        )

        # Document distribution
        doc_sections = list(self.analytics_data.section_coverage.keys())
        doc_counts = [
            len([doc for doc in self.discovery_system.documents.values()
                 if doc['section'] == section])
            for section in doc_sections
        ]

        fig.add_trace(
            go.Pie(
                labels=doc_sections,
                values=doc_counts,
                name="Documents by Section"
            ),
            row=1, col=1
        )

        # Topic distribution
        topic_names = list(self.analytics_data.topic_distribution.keys())
        topic_counts = list(self.analytics_data.topic_distribution.values())

        fig.add_trace(
            go.Bar(
                x=topic_names,
                y=topic_counts,
                marker_color='lightblue',
                name="Topics"
            ),
            row=1, col=2
        )

        # Difficulty levels
        diff_levels = list(self.analytics_data.difficulty_distribution.keys())
        diff_counts = list(self.analytics_data.difficulty_distribution.values())

        fig.add_trace(
            go.Bar(
                x=diff_levels,
                y=diff_counts,
                marker_color='lightgreen',
                name="Difficulty"
            ),
            row=1, col=3
        )

        # Section coverage
        coverage_sections = list(self.analytics_data.section_coverage.keys())
        coverage_values = list(self.analytics_data.section_coverage.values())

        fig.add_trace(
            go.Bar(
                x=coverage_sections,
                y=coverage_values,
                marker_color='orange',
                name="Coverage %"
            ),
            row=2, col=1
        )

        # Cross-reference density indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self.analytics_data.cross_reference_density * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "X-Ref Density %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )

        # Trending topics
        trending_names = [topic[0] for topic in self.analytics_data.trending_topics]
        trending_scores = [topic[1] for topic in self.analytics_data.trending_topics]

        fig.add_trace(
            go.Bar(
                x=trending_names,
                y=trending_scores,
                marker_color='purple',
                name="Trending Score"
            ),
            row=2, col=3
        )

        # Knowledge gaps by type
        gap_types = [gap['gap_type'] for gap in self.analytics_data.knowledge_gaps]
        gap_counts = dict(Counter(gap_types))

        fig.add_trace(
            go.Bar(
                x=list(gap_counts.keys()),
                y=list(gap_counts.values()),
                marker_color='red',
                name="Knowledge Gaps"
            ),
            row=3, col=1
        )

        # Content growth over time (mock data)
        dates = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05']
        content_counts = [100, 120, 135, 150, 165]

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=content_counts,
                mode='lines+markers',
                name="Content Count"
            ),
            row=3, col=2
        )

        # Learning path completion (mock data)
        paths = ['Beginner ML', 'Deep Learning', 'NLP Basics', 'CV Intro', 'AI Ethics']
        completion_rates = [85, 60, 45, 30, 70]

        fig.add_trace(
            go.Bar(
                x=paths,
                y=completion_rates,
                marker_color='teal',
                name="Completion %"
            ),
            row=3, col=3
        )

        fig.update_layout(
            title="Knowledge Graph Analytics Dashboard",
            height=1200,
            showlegend=False
        )

        return fig

    def _generate_analytics_data(self):
        """Generate analytics data for dashboard"""
        # Get document statistics
        sections = [doc['section'] for doc in self.discovery_system.documents.values()]
        section_counts = Counter(sections)

        # Calculate section coverage
        section_coverage = {}
        for section in set(sections):
            total_docs = len(self.discovery_system.documents)
            section_docs = section_counts[section]
            section_coverage[section] = (section_docs / total_docs) * 100

        # Get topic distribution
        topic_distribution = {
            topic.name: len(topic.documents)
            for topic in self.discovery_system.topics.values()
        }

        # Get difficulty distribution
        difficulties = [
            self.discovery_system._estimate_document_difficulty(doc)
            for doc in self.discovery_system.documents.values()
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

        # Calculate cross-reference density
        total_possible_refs = len(self.discovery_system.documents) ** 2
        actual_refs = len(self.cross_referencer.cross_references)
        cross_ref_density = actual_refs / total_possible_refs if total_possible_refs > 0 else 0

        # Get trending topics
        trending_topics = self.discovery_system.detect_trending_topics()
        trending_list = [
            (self.discovery_system.topics[tid].name, score)
            for tid, score in sorted(trending_topics.items(), key=lambda x: x[1], reverse=True)
        ][:10]

        # Get knowledge gaps
        knowledge_gaps = [
            {'gap_type': gap.gap_type, 'description': gap.description}
            for gap in self.discovery_system.content_gaps.values()
        ]

        self.analytics_data = AnalyticsData(
            total_concepts=len(self.knowledge_graph.concepts),
            total_relationships=len(self.knowledge_graph.relationships),
            total_documents=len(self.discovery_system.documents),
            topic_distribution=topic_distribution,
            difficulty_distribution=difficulty_bins,
            section_coverage=section_coverage,
            cross_reference_density=cross_ref_density,
            learning_progress={},  # Would be populated with user data
            trending_topics=trending_list,
            knowledge_gaps=knowledge_gaps
        )

    def create_interactive_dashboard(self) -> go.Figure:
        """Create main interactive dashboard with multiple views"""
        logger.info("Creating interactive dashboard...")

        # Create tabs for different views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Knowledge Graph Overview',
                'Content Distribution',
                'Learning Progress',
                'System Analytics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )

        # Knowledge graph overview (simplified)
        kg_stats = self.knowledge_graph.get_system_statistics()
        fig.add_trace(
            go.Scatter(
                x=[1, 2, 3, 4],
                y=[
                    kg_stats['concepts']['total'],
                    kg_stats['relationships']['total'],
                    len(self.discovery_system.topics),
                    len(self.discovery_system.clusters)
                ],
                mode='markers+text',
                text=['Concepts', 'Relationships', 'Topics', 'Clusters'],
                textposition='top center',
                marker=dict(size=[30, 25, 20, 15], color=['blue', 'green', 'orange', 'red']),
                name='Knowledge Graph Components'
            ),
            row=1, col=1
        )

        # Content distribution pie chart
        sections = [doc['section'] for doc in self.discovery_system.documents.values()]
        section_counts = Counter(sections)

        fig.add_trace(
            go.Pie(
                labels=list(section_counts.keys()),
                values=list(section_counts.values()),
                name="Content by Section"
            ),
            row=1, col=2
        )

        # Learning progress (mock data)
        learning_areas = ['Theory', 'Practice', 'Projects', 'Assessments']
        progress_scores = [75, 60, 45, 80]

        fig.add_trace(
            go.Bar(
                x=learning_areas,
                y=progress_scores,
                marker_color='lightblue',
                name="Progress %"
            ),
            row=2, col=1
        )

        # System health indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=85,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )

        fig.update_layout(
            title="AI Documentation Knowledge Dashboard",
            height=800,
            showlegend=False
        )

        return fig

    def save_visualization(self, fig: go.Figure, filename: str, format: str = 'html'):
        """Save visualization to file"""
        output_path = self.base_path / f"visualizations/{filename}"

        # Create visualizations directory if it doesn't exist
        output_path.parent.mkdir(exist_ok=True)

        if format == 'html':
            fig.write_html(str(output_path))
        elif format == 'png':
            fig.write_image(str(output_path))
        elif format == 'svg':
            fig.write_image(str(output_path), format='svg')
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Visualization saved to {output_path}")

    def generate_all_visualizations(self):
        """Generate all standard visualizations"""
        logger.info("Generating all visualizations...")

        # Ensure data is available
        self.knowledge_graph.build_knowledge_graph()
        self.discovery_system.discover_topics()
        self.discovery_system.cluster_content()

        # Generate visualizations
        visualizations = {
            'knowledge_graph': self.create_interactive_knowledge_graph(),
            'topic_visualization': self.create_topic_visualization(),
            'learning_dashboard': self.create_learning_progress_dashboard(),
            'content_heatmap': self.create_content_heatmap(),
            'concept_space_3d': self.create_3d_concept_space(),
            'cross_reference_network': self.create_cross_reference_network(),
            'analytics_dashboard': self.create_analytics_dashboard(),
            'interactive_dashboard': self.create_interactive_dashboard()
        }

        # Save all visualizations
        for name, fig in visualizations.items():
            self.save_visualization(fig, f"{name}.html")

        logger.info("All visualizations generated and saved!")

        return visualizations

def main():
    """Main function to run the visualization system"""
    # Initialize the system
    viz_system = KnowledgeVisualizationSystem()

    # Generate all visualizations
    visualizations = viz_system.generate_all_visualizations()

    print("Knowledge Visualization System Results:")
    print(f"Generated {len(visualizations)} visualizations:")
    for name in visualizations.keys():
        print(f"  - {name}")

    # Create a summary report
    summary = {
        'generated_visualizations': list(visualizations.keys()),
        'total_concepts': len(viz_system.knowledge_graph.concepts),
        'total_relationships': len(viz_system.knowledge_graph.relationships),
        'total_documents': len(viz_system.discovery_system.documents),
        'discovered_topics': len(viz_system.discovery_system.topics),
        'content_clusters': len(viz_system.discovery_system.clusters),
        'knowledge_gaps': len(viz_system.discovery_system.content_gaps),
        'trending_topics': len(viz_system.discovery_system.trending_topics),
        'generation_timestamp': datetime.now().isoformat()
    }

    with open('/Users/dtumkorkmaz/Projects/ai-docs/visualization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nVisualization summary saved to visualization_summary.json")
    print("All visualizations are available in the 'visualizations' directory!")

if __name__ == "__main__":
    main()