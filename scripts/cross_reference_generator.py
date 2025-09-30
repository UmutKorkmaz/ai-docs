#!/usr/bin/env python3
"""
AI Documentation Cross-Reference Generator
Automatically creates intelligent links between related concepts across documentation sections.
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict
import networkx as nx
from difflib import SequenceMatcher

@dataclass
class Concept:
    """Represents a concept in the AI documentation"""
    id: str
    name: str
    section: str
    category: str
    difficulty: str
    description: str = ""
    content: str = ""
    file_path: str = ""
    prerequisites: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    code_patterns: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class Relationship:
    """Represents a relationship between two concepts"""
    source: str
    target: str
    type: str
    strength: float
    description: str = ""

class CrossReferenceGenerator:
    """Main class for generating cross-references"""

    def __init__(self, docs_root: str):
        self.docs_root = Path(docs_root)
        self.concepts: Dict[str, Concept] = {}
        self.relationships: List[Relationship] = []
        self.knowledge_graph = nx.DiGraph()

        # Configuration
        self.similarity_threshold = 0.3
        self.code_pattern_similarity = 0.5
        self.min_relationship_strength = 0.2

    def extract_concepts(self) -> None:
        """Extract concepts from all documentation files"""
        print("Extracting concepts from documentation...")

        # Supported file extensions
        extensions = ['.md', '.py', '.ipynb']

        for ext in extensions:
            for file_path in self.docs_root.rglob(f'*{ext}'):
                self._process_file(file_path)

        print(f"Extracted {len(self.concepts)} concepts")

    def _process_file(self, file_path: Path) -> None:
        """Process a single file to extract concepts"""
        try:
            content = file_path.read_text(encoding='utf-8')

            # Skip if file is too small
            if len(content) < 100:
                return

            # Extract section from path
            section = self._get_section_from_path(file_path)

            # Identify concepts in file
            concepts_in_file = self._identify_concepts(content, section, str(file_path))

            for concept in concepts_in_file:
                self.concepts[concept.id] = concept

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def _get_section_from_path(self, file_path: Path) -> str:
        """Extract section name from file path"""
        # Get parent directory name
        for parent in file_path.parents:
            if parent.name.isdigit() and '_' in parent.name:
                return parent.name
        return "misc"

    def _identify_concepts(self, content: str, section: str, file_path: str) -> List[Concept]:
        """Identify concepts in file content"""
        concepts = []

        # Look for headings and key terms
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)

        # Extract code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', content, re.DOTALL)

        # For each major heading, create a concept
        for i, heading in enumerate(headings):
            # Clean heading
            clean_heading = re.sub(r'[*`]', '', heading).strip()

            if len(clean_heading) < 3:
                continue

            # Generate concept ID
            concept_id = f"{section}_{clean_heading.lower().replace(' ', '_').replace('-', '_')}"

            # Get relevant content
            start_idx = content.find(f"# {heading}")
            next_heading_idx = len(content)
            for h in headings[i+1:]:
                idx = content.find(f"# {h}")
                if idx > start_idx:
                    next_heading_idx = idx
                    break

            concept_content = content[start_idx:next_heading_idx].strip()

            # Create concept
            concept = Concept(
                id=concept_id,
                name=clean_heading,
                section=section,
                category="theory" if "theory" in file_path.lower() else "implementation",
                difficulty=self._estimate_difficulty(concept_content),
                content=concept_content,
                file_path=file_path,
                code_patterns=code_blocks
            )

            # Extract tags
            concept.tags = self._extract_tags(concept_content)

            concepts.append(concept)

        return concepts

    def _estimate_difficulty(self, content: str) -> str:
        """Estimate difficulty level based on content"""
        # Simple heuristics
        advanced_keywords = [
            'quantum', 'neuromorphic', 'transformer', 'gan', 'reinforcement',
            'attention', 'embeddings', 'backpropagation', 'gradient descent',
            'convolution', 'recurrent', 'lstm', 'gru'
        ]

        expert_keywords = [
            'variational', 'bayesian inference', 'markov chain',
            'stochastic', 'optimization', 'theoretical', 'proof'
        ]

        content_lower = content.lower()

        if any(kw in content_lower for kw in expert_keywords):
            return "expert"
        elif any(kw in content_lower for kw in advanced_keywords):
            return "advanced"
        elif len(content) > 2000:
            return "intermediate"
        else:
            return "beginner"

    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from content"""
        tags = []

        # Common AI tags
        common_tags = [
            'machine learning', 'deep learning', 'neural networks',
            'natural language processing', 'computer vision',
            'reinforcement learning', 'generative ai',
            'ethics', 'safety', 'applications'
        ]

        content_lower = content.lower()
        for tag in common_tags:
            if tag in content_lower:
                tags.append(tag)

        return tags

    def discover_relationships(self) -> None:
        """Discover relationships between concepts"""
        print("Discovering relationships...")

        # Semantic similarity
        self._discover_semantic_relationships()

        # Code pattern similarity
        self._discover_code_relationships()

        # Prerequisite relationships
        self._discover_prerequisite_relationships()

        # Application relationships
        self._discover_application_relationships()

        # Build knowledge graph
        self._build_knowledge_graph()

        print(f"Discovered {len(self.relationships)} relationships")

    def _discover_semantic_relationships(self) -> None:
        """Discover relationships based on semantic similarity"""
        concept_list = list(self.concepts.values())

        for i, concept1 in enumerate(concept_list):
            for concept2 in concept_list[i+1:]:
                # Calculate similarity
                similarity = self._calculate_semantic_similarity(
                    concept1.content, concept2.content
                )

                if similarity > self.similarity_threshold:
                    # Determine relationship type
                    rel_type = self._determine_relationship_type(
                        concept1, concept2, similarity
                    )

                    if rel_type:
                        self.relationships.append(Relationship(
                            source=concept1.id,
                            target=concept2.id,
                            type=rel_type,
                            strength=similarity,
                            description=f"Semantic similarity: {similarity:.2f}"
                        ))

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        # Simple implementation using word overlap
        # In production, use BERT or similar embeddings

        # Extract words
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _determine_relationship_type(self, concept1: Concept, concept2: Concept,
                                   similarity: float) -> Optional[str]:
        """Determine relationship type based on concept properties"""

        # Check if one is implementation of other
        if concept1.category == "theory" and concept2.category == "implementation":
            return "IMPLEMENTS"
        elif concept1.category == "implementation" and concept2.category == "theory":
            return "IMPLEMENTED_BY"

        # Check difficulty progression
        difficulty_order = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
        if difficulty_order.get(concept1.difficulty, 0) < difficulty_order.get(concept2.difficulty, 0):
            return "BUILDS_ON"
        elif difficulty_order.get(concept1.difficulty, 0) > difficulty_order.get(concept2.difficulty, 0):
            return "EXTENDS"

        # High similarity suggests related concepts
        if similarity > 0.7:
            return "SIMILAR_TO"
        elif similarity > 0.5:
            return "RELATED_TO"

        return None

    def _discover_code_relationships(self) -> None:
        """Discover relationships based on code patterns"""
        # Group concepts by code patterns
        pattern_to_concepts = defaultdict(list)

        for concept in self.concepts.values():
            for pattern in concept.code_patterns:
                pattern_hash = hash(pattern)
                pattern_to_concepts[pattern_hash].append(concept.id)

        # Create relationships for concepts with similar code
        for pattern_hash, concept_ids in pattern_to_concepts.items():
            if len(concept_ids) > 1:
                for i, id1 in enumerate(concept_ids):
                    for id2 in concept_ids[i+1:]:
                        self.relationships.append(Relationship(
                            source=id1,
                            target=id2,
                            type="SIMILAR_IMPLEMENTATION",
                            strength=0.8,
                            description="Shares code patterns"
                        ))

    def _discover_prerequisite_relationships(self) -> None:
        """Discover prerequisite relationships based on content analysis"""
        # Look for prerequisite indicators in content
        prerequisite_indicators = [
            'requires knowledge of', 'assumes familiarity with',
            'builds on', 'prerequisite', 'before reading'
        ]

        for concept in self.concepts.values():
            content_lower = concept.content.lower()

            for indicator in prerequisite_indicators:
                if indicator in content_lower:
                    # Find mentioned concepts
                    for other_id, other_concept in self.concepts.items():
                        if other_id != concept.id:
                            if other_concept.name.lower() in content_lower:
                                self.relationships.append(Relationship(
                                    source=other_id,
                                    target=concept.id,
                                    type="PREREQUISITE_OF",
                                    strength=0.9,
                                    description=f"Mentioned as {indicator}"
                                ))

    def _discover_application_relationships(self) -> None:
        """Discover application relationships"""
        application_keywords = [
            'application', 'use case', 'applied to', 'used in',
            'practical example', 'real-world'
        ]

        for concept in self.concepts.values():
            content_lower = concept.content.lower()

            for keyword in application_keywords:
                if keyword in content_lower:
                    # Find applied concepts
                    for other_id, other_concept in self.concepts.items():
                        if other_id != concept.id:
                            if other_concept.name.lower() in content_lower:
                                self.relationships.append(Relationship(
                                    source=other_id,
                                    target=concept.id,
                                    type="APPLICATION_OF",
                                    strength=0.7,
                                    description=f"Applied {keyword}"
                                ))

    def _build_knowledge_graph(self) -> None:
        """Build NetworkX knowledge graph"""
        # Add nodes
        for concept_id, concept in self.concepts.items():
            self.knowledge_graph.add_node(
                concept_id,
                name=concept.name,
                section=concept.section,
                category=concept.category,
                difficulty=concept.difficulty,
                tags=concept.tags
            )

        # Add edges
        for rel in self.relationships:
            self.knowledge_graph.add_edge(
                rel.source,
                rel.target,
                type=rel.type,
                strength=rel.strength,
                description=rel.description
            )

    def generate_learning_paths(self) -> Dict[str, List[str]]:
        """Generate learning paths based on prerequisites"""
        print("Generating learning paths...")

        learning_paths = {}

        # Beginner path
        beginner_concepts = [
            c for c in self.concepts.values()
            if c.difficulty == "beginner"
        ]
        learning_paths["beginner"] = self._topological_sort(
            [c.id for c in beginner_concepts]
        )

        # Specialist paths by section
        sections = set(c.section for c in self.concepts.values())
        for section in sections:
            section_concepts = [
                c for c in self.concepts.values()
                if c.section == section
            ]
            learning_paths[f"{section}_specialist"] = self._topological_sort(
                [c.id for c in section_concepts]
            )

        return learning_paths

    def _topological_sort(self, concept_ids: List[str]) -> List[str]:
        """Topological sort considering prerequisites"""
        # Create subgraph
        subgraph = self.knowledge_graph.subgraph(concept_ids)

        try:
            return list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            # If cycle exists, return original order
            return concept_ids

    def generate_cross_reference_links(self) -> Dict[str, List[Dict]]:
        """Generate cross-reference links for each concept"""
        print("Generating cross-reference links...")

        links = {}

        for concept_id, concept in self.concepts.items():
            concept_links = []

            # Get all relationships for this concept
            incoming = list(self.knowledge_graph.in_edges(concept_id, data=True))
            outgoing = list(self.knowledge_graph.out_edges(concept_id, data=True))

            # Process relationships
            for source, target, data in incoming + outgoing:
                related_id = target if source == concept_id else source
                related_concept = self.concepts[related_id]

                # Determine link direction
                is_outgoing = source == concept_id

                concept_links.append({
                    "concept_id": related_id,
                    "concept_name": related_concept.name,
                    "relationship_type": data["type"],
                    "relationship_strength": data["strength"],
                    "direction": "outgoing" if is_outgoing else "incoming",
                    "section": related_concept.section,
                    "difficulty": related_concept.difficulty
                })

            # Sort by strength
            concept_links.sort(key=lambda x: x["relationship_strength"], reverse=True)

            links[concept_id] = concept_links[:10]  # Top 10 links

        return links

    def export_results(self, output_dir: str) -> None:
        """Export all results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export concepts
        with open(output_path / "concepts.json", "w") as f:
            json.dump({
                id: {
                    "id": c.id,
                    "name": c.name,
                    "section": c.section,
                    "category": c.category,
                    "difficulty": c.difficulty,
                    "tags": c.tags,
                    "file_path": c.file_path
                }
                for id, c in self.concepts.items()
            }, f, indent=2)

        # Export relationships
        with open(output_path / "relationships.json", "w") as f:
            json.dump([
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.type,
                    "strength": r.strength,
                    "description": r.description
                }
                for r in self.relationships
            ], f, indent=2)

        # Export learning paths
        learning_paths = self.generate_learning_paths()
        with open(output_path / "learning_paths.json", "w") as f:
            json.dump(learning_paths, f, indent=2)

        # Export cross-reference links
        links = self.generate_cross_reference_links()
        with open(output_path / "cross_reference_links.json", "w") as f:
            json.dump(links, f, indent=2)

        # Export graph for visualization
        nx.write_gexf(self.knowledge_graph, output_path / "knowledge_graph.gexf")

        print(f"Results exported to {output_path}")

def main():
    """Main execution function"""
    docs_root = "/Users/dtumkorkmaz/Projects/ai-docs"
    output_dir = "/Users/dtumkorkmaz/Projects/ai-docs/cross_reference_output"

    # Initialize generator
    generator = CrossReferenceGenerator(docs_root)

    # Extract concepts
    generator.extract_concepts()

    # Discover relationships
    generator.discover_relationships()

    # Export results
    generator.export_results(output_dir)

    print("Cross-reference generation complete!")

if __name__ == "__main__":
    main()