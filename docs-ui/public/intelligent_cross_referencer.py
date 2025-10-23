#!/usr/bin/env python3
"""
Intelligent Cross-Referencing System for AI Documentation

This system provides:
- Automatic cross-reference discovery and insertion
- Context-aware link suggestions based on content analysis
- Bidirectional relationship mapping
- Dynamic cross-reference updates as content evolves
- Smart link placement and formatting
- Cross-reference validation and maintenance
"""

import json
import re
import os
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from collections import defaultdict, Counter

import networkx as nx
import numpy as np

# Import knowledge graph system
from knowledge_graph_system import AIKnowledgeGraph, AIConcept, ContentRelationship

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CrossReference:
    """Represents a cross-reference between content pieces"""
    id: str
    source_file: str
    target_file: str
    source_text: str
    target_concept: str
    reference_type: str  # 'definition', 'example', 'application', 'related', 'contrast'
    context_before: str
    context_after: str
    confidence_score: float  # 0.0-1.0
    auto_generated: bool = True
    created_at: datetime = datetime.now()
    validated: bool = False

@dataclass
class LinkSuggestion:
    """Represents a suggested link to add"""
    text_to_link: str
    target_file: str
    target_section: str
    link_text: str
    reason: str
    confidence: float
    position_in_text: int
    surrounding_context: str

class IntelligentCrossReferencer:
    """Main cross-referencing system"""

    def __init__(self, base_path: str = "/Users/dtumkorkmaz/Projects/ai-docs"):
        self.base_path = Path(base_path)
        self.knowledge_graph = AIKnowledgeGraph(base_path)

        # Cross-reference configuration
        self.config = {
            'min_confidence_threshold': 0.3,
            'max_suggestions_per_file': 20,
            'context_window_size': 100,  # characters
            'auto_insert_threshold': 0.8,
            'link_types': {
                'definition': ['definition', 'means', 'refers to', 'is defined as'],
                'example': ['example', 'for instance', 'such as', 'like'],
                'application': ['application', 'use case', 'applied in', 'used for'],
                'related': ['related', 'similar', 'connected', 'associated'],
                'contrast': ['contrast', 'difference', 'unlike', 'different from']
            },
            'excluded_patterns': [
                r'^\s*#',  # Headers
                r'^\s*```',  # Code blocks
                r'^\s*>',  # Blockquotes
                r'^\s*-',  # List items
                r'^\s*\d+\.',  # Numbered lists
            ]
        }

        # Initialize data structures
        self.cross_references: Dict[str, CrossReference] = {}
        self.file_index: Dict[str, Dict] = {}
        self.content_cache: Dict[str, str] = {}
        self.link_patterns: Dict[str, List[str]] = self._initialize_link_patterns()

        # Load existing data
        self._load_existing_data()

    def _initialize_link_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for different types of links"""
        return {
            'internal_concept': [
                r'\b({concept})\b',
                r'\b({concept}s?)\b',
                r'\b({concept} learning)\b',
                r'\b({concept} networks?)\b',
            ],
            'method_reference': [
                r'\b(using|with|via) ({concept})\b',
                r'\b(based on|using) ({concept})\b',
                r'\b({concept}) (method|algorithm|approach)\b',
            ],
            'application_mention': [
                r'\b({concept}) (in|for|with) ({context})\b',
                r'\b({context}) (using|with) ({concept})\b',
            ],
            'comparison': [
                r'\b({concept}) (vs|versus|compared to) ({other})\b',
                r'\b(unlike|different from) ({concept})\b',
            ]
        }

    def _load_existing_data(self):
        """Load existing cross-reference data"""
        cache_file = self.base_path / "cross_reference_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.cross_references = {
                        ref_id: CrossReference(**ref_data)
                        for ref_id, ref_data in data.get('cross_references', {}).items()
                    }
                    self.file_index = data.get('file_index', {})
                logger.info(f"Loaded {len(self.cross_references)} existing cross-references")
            except Exception as e:
                logger.warning(f"Could not load cross-reference cache: {e}")

    def _save_cache(self):
        """Save current cross-reference data"""
        cache_file = self.base_path / "cross_reference_cache.json"
        try:
            data = {
                'cross_references': {
                    ref_id: asdict(ref) for ref_id, ref in self.cross_references.items()
                },
                'file_index': self.file_index,
                'last_updated': datetime.now().isoformat()
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Cross-reference cache saved")
        except Exception as e:
            logger.error(f"Could not save cache: {e}")

    def index_all_files(self):
        """Index all markdown files for cross-referencing"""
        logger.info("Indexing files for cross-referencing...")

        md_files = list(self.base_path.glob("**/*.md"))

        for file_path in md_files:
            self._index_file(file_path)

        logger.info(f"Indexed {len(md_files)} files")

    def _index_file(self, file_path: Path):
        """Index a single file for cross-referencing"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse content into sections
            sections = self._parse_markdown_sections(content)

            # Extract key concepts and terms
            concepts = self._extract_key_concepts(content)

            # Store in index
            self.file_index[str(file_path)] = {
                'sections': sections,
                'concepts': concepts,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'size': len(content)
            }

            # Cache content
            self.content_cache[str(file_path)] = content

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")

    def _parse_markdown_sections(self, content: str) -> List[Dict]:
        """Parse markdown content into sections"""
        sections = []
        lines = content.split('\n')
        current_section = None

        for i, line in enumerate(lines):
            # Check for headers
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
            elif current_section:
                current_section['content'].append(line)

        # Add the last section
        if current_section:
            sections.append(current_section)

        return sections

    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        concepts = []

        # Use knowledge graph concepts
        for concept_id, concept in self.knowledge_graph.concepts.items():
            if concept.name.lower() in content.lower():
                concepts.append({
                    'id': concept_id,
                    'name': concept.name,
                    'section': concept.section,
                    'type': concept.concept_type
                })

        # Extract additional concepts using patterns
        ai_concept_patterns = [
            r'\b(supervised|unsupervised|reinforcement)\s+learning\b',
            r'\b(deep|neural|artificial)\s+(network|intelligence)\b',
            r'\b(machine|natural\s+language|computer)\s+(learning|processing|vision)\b',
            r'\b(transformer|convolutional|recurrent)\s+(network|model)\b',
            r'\b(generative|discriminative|adversarial)\s+(network|model)\b',
        ]

        for pattern in ai_concept_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                concept = match.group().lower()
                if concept not in [c['name'] for c in concepts]:
                    concepts.append({
                        'id': f"extracted_{hash(concept)}",
                        'name': concept,
                        'section': 'extracted',
                        'type': 'pattern'
                    })

        return concepts

    def discover_cross_references(self, file_path: str) -> List[CrossReference]:
        """Discover cross-references for a given file"""
        if file_path not in self.file_index:
            self._index_file(Path(file_path))

        file_data = self.file_index[file_path]
        content = self.content_cache[file_path]
        cross_refs = []

        # Discover references to concepts in other files
        for concept in file_data['concepts']:
            if concept['id'] in self.knowledge_graph.concepts:
                kg_concept = self.knowledge_graph.concepts[concept['id']]

                # Find mentions in current file
                mentions = self._find_concept_mentions(content, kg_concept.name)

                for mention in mentions:
                    if kg_concept.file_path != file_path:  # Don't reference same file
                        cross_ref = self._create_cross_reference(
                            file_path, kg_concept, mention, content
                        )
                        if cross_ref.confidence_score >= self.config['min_confidence_threshold']:
                            cross_refs.append(cross_ref)

        # Discover related content references
        related_refs = self._discover_related_content_references(file_path, content)
        cross_refs.extend(related_refs)

        # Remove duplicates and sort by confidence
        cross_refs = self._deduplicate_cross_references(cross_refs)
        cross_refs.sort(key=lambda x: x.confidence_score, reverse=True)

        return cross_refs

    def _find_concept_mentions(self, content: str, concept_name: str) -> List[Dict]:
        """Find mentions of a concept in content"""
        mentions = []
        lines = content.split('\n')

        # Create various patterns to match the concept
        patterns = [
            rf'\b{re.escape(concept_name)}\b',
            rf'\b{re.escape(concept_name)}s?\b',
            rf'\b{re.escape(concept_name)}\s+(learning|network|model|algorithm)\b',
        ]

        for line_num, line in enumerate(lines):
            # Skip excluded patterns
            if any(re.match(pattern, line) for pattern in self.config['excluded_patterns']):
                continue

            for pattern in patterns:
                matches = list(re.finditer(pattern, line, re.IGNORECASE))
                for match in matches:
                    mentions.append({
                        'line_number': line_num,
                        'column_start': match.start(),
                        'column_end': match.end(),
                        'matched_text': match.group(),
                        'full_line': line,
                        'context': self._get_context(content, line_num, match.start())
                    })

        return mentions

    def _get_context(self, content: str, line_num: int, col_pos: int) -> Dict:
        """Get context around a mention"""
        lines = content.split('\n')

        # Get before and after text
        context_window = self.config['context_window_size']
        line = lines[line_num]

        start_pos = max(0, col_pos - context_window)
        end_pos = min(len(line), col_pos + context_window + len(line) - col_pos)

        context_before = line[start_pos:col_pos]
        context_after = line[col_pos:end_pos]

        # Get surrounding lines
        surrounding_lines = []
        for i in range(max(0, line_num - 2), min(len(lines), line_num + 3)):
            if i != line_num:
                surrounding_lines.append(lines[i])

        return {
            'context_before': context_before,
            'context_after': context_after,
            'surrounding_lines': surrounding_lines,
            'full_context': ' '.join(surrounding_lines)
        }

    def _create_cross_reference(self, source_file: str, target_concept: AIConcept,
                               mention: Dict, content: str) -> CrossReference:
        """Create a cross-reference object"""
        # Determine reference type
        ref_type = self._determine_reference_type(mention['context'])

        # Calculate confidence score
        confidence = self._calculate_confidence_score(mention, target_concept, ref_type)

        cross_ref = CrossReference(
            id=f"{hashlib.md5(f'{source_file}_{target_concept.id}_{mention["line_number"]}'.encode()).hexdigest()[:8]}",
            source_file=source_file,
            target_file=target_concept.file_path,
            source_text=mention['matched_text'],
            target_concept=target_concept.name,
            reference_type=ref_type,
            context_before=mention['context']['context_before'],
            context_after=mention['context']['context_after'],
            confidence_score=confidence
        )

        return cross_ref

    def _determine_reference_type(self, context: Dict) -> str:
        """Determine the type of reference based on context"""
        text = (context['context_before'] + ' ' + context['context_after']).lower()

        for ref_type, indicators in self.config['link_types'].items():
            if any(indicator in text for indicator in indicators):
                return ref_type

        return 'related'  # Default type

    def _calculate_confidence_score(self, mention: Dict, target_concept: AIConcept,
                                  ref_type: str) -> float:
        """Calculate confidence score for a cross-reference"""
        score = 0.0

        # Base score for exact match
        score += 0.4

        # Contextual clues
        context_text = (mention['context']['context_before'] + ' ' +
                       mention['context']['context_after']).lower()

        # Strong indicators for different reference types
        type_scores = {
            'definition': 0.3,
            'example': 0.25,
            'application': 0.2,
            'related': 0.15,
            'contrast': 0.2
        }
        score += type_scores.get(ref_type, 0.1)

        # Position in document (earlier mentions might be more important)
        if mention['line_number'] < 50:  # Early in document
            score += 0.1

        # Length of matched text (longer matches are more specific)
        if len(mention['matched_text']) > 10:
            score += 0.1

        # Concept difficulty (more advanced concepts might need more references)
        if target_concept.difficulty_level >= 3:
            score += 0.05

        return min(score, 1.0)

    def _discover_related_content_references(self, file_path: str, content: str) -> List[CrossReference]:
        """Discover references to related content using semantic similarity"""
        related_refs = []

        # Use knowledge graph relationships
        for relationship in self.knowledge_graph.relationships:
            source_file = self.knowledge_graph.concepts[relationship.source_id].file_path
            target_file = self.knowledge_graph.concepts[relationship.target_id].file_path

            if source_file == file_path and target_file != file_path:
                # Find where this relationship is mentioned
                target_concept = self.knowledge_graph.concepts[relationship.target_id]
                mentions = self._find_concept_mentions(content, target_concept.name)

                for mention in mentions:
                    cross_ref = CrossReference(
                        id=f"rel_{hashlib.md5(f'{file_path}_{target_concept.id}_{mention["line_number"]}'.encode()).hexdigest()[:8]}",
                        source_file=file_path,
                        target_file=target_file,
                        source_text=mention['matched_text'],
                        target_concept=target_concept.name,
                        reference_type='related',
                        context_before=mention['context']['context_before'],
                        context_after=mention['context']['context_after'],
                        confidence_score=relationship.strength * 0.8  # Slightly lower for indirect refs
                    )
                    related_refs.append(cross_ref)

        return related_refs

    def _deduplicate_cross_references(self, cross_refs: List[CrossReference]) -> List[CrossReference]:
        """Remove duplicate cross-references"""
        seen = set()
        unique_refs = []

        for ref in cross_refs:
            # Create a key for deduplication
            key = (ref.source_file, ref.target_file, ref.source_text.lower())
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)

        return unique_refs

    def generate_link_suggestions(self, file_path: str, max_suggestions: int = None) -> List[LinkSuggestion]:
        """Generate link suggestions for a file"""
        if max_suggestions is None:
            max_suggestions = self.config['max_suggestions_per_file']

        cross_refs = self.discover_cross_references(file_path)
        suggestions = []

        for cross_ref in cross_refs[:max_suggestions]:
            # Create link suggestion
            suggestion = LinkSuggestion(
                text_to_link=cross_ref.source_text,
                target_file=cross_ref.target_file,
                target_section=self.knowledge_graph.concepts.get(
                    next((cid for cid, concept in self.knowledge_graph.concepts.items()
                         if concept.name == cross_ref.target_concept), None)
                ).section if self.knowledge_graph.concepts else "Unknown",
                link_text=f"[{cross_ref.source_text}]({cross_ref.target_file})",
                reason=f"Reference to {cross_ref.target_concept} ({cross_ref.reference_type})",
                confidence=cross_ref.confidence_score,
                position_in_text=0,  # Would need to calculate actual position
                surrounding_context=f"{cross_ref.context_before}...{cross_ref.context_after}"
            )
            suggestions.append(suggestion)

        return suggestions

    def insert_cross_references(self, file_path: str, auto_insert: bool = False) -> Dict:
        """Insert cross-references into a file"""
        if file_path not in self.content_cache:
            self._index_file(Path(file_path))

        content = self.content_cache[file_path]
        suggestions = self.generate_link_suggestions(file_path)

        inserted_count = 0
        modified_content = content
        insertions = []

        # Sort suggestions by position (reverse order to maintain offsets)
        suggestions.sort(key=lambda x: x.position_in_text, reverse=True)

        for suggestion in suggestions:
            # Only auto-insert high-confidence suggestions
            if auto_insert and suggestion.confidence >= self.config['auto_insert_threshold']:
                # Find the text to replace
                text_pattern = re.escape(suggestion.text_to_link)

                # Replace with markdown link
                modified_content = re.sub(
                    text_pattern,
                    suggestion.link_text,
                    modified_content,
                    count=1
                )

                insertions.append({
                    'text': suggestion.text_to_link,
                    'link': suggestion.link_text,
                    'confidence': suggestion.confidence,
                    'reason': suggestion.reason
                })
                inserted_count += 1

        # Save modified content if changes were made
        if auto_insert and inserted_count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            logger.info(f"Inserted {inserted_count} cross-references in {file_path}")

        return {
            'file_path': file_path,
            'total_suggestions': len(suggestions),
            'inserted_count': inserted_count,
            'insertions': insertions,
            'remaining_suggestions': len(suggestions) - inserted_count
        }

    def validate_cross_references(self, file_path: str = None) -> Dict[str, List[str]]:
        """Validate existing cross-references"""
        validation_results = {
            'broken_links': [],
            'missing_targets': [],
            'invalid_format': []
        }

        files_to_check = [file_path] if file_path else list(self.content_cache.keys())

        for current_file in files_to_check:
            if current_file not in self.content_cache:
                continue

            content = self.content_cache[current_file]

            # Find all markdown links
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            matches = re.finditer(link_pattern, content)

            for match in matches:
                link_text = match.group(1)
                target_path = match.group(2)

                # Check if target file exists
                if not target_path.startswith('http'):  # Skip external links
                    target_full_path = self.base_path / target_path

                    if not target_full_path.exists():
                        validation_results['broken_links'].append({
                            'source_file': current_file,
                            'link_text': link_text,
                            'target_path': target_path,
                            'line': content[:match.start()].count('\n') + 1
                        })
                    elif target_full_path.suffix != '.md':
                        # Link to non-markdown file
                        validation_results['invalid_format'].append({
                            'source_file': current_file,
                            'link_text': link_text,
                            'target_path': target_path,
                            'reason': 'Target is not a markdown file'
                        })

        return validation_results

    def generate_cross_reference_report(self) -> Dict:
        """Generate comprehensive cross-reference report"""
        # Index all files if not already done
        if not self.file_index:
            self.index_all_files()

        # Discover cross-references for all files
        all_cross_refs = []
        for file_path in self.content_cache.keys():
            cross_refs = self.discover_cross_references(file_path)
            all_cross_refs.extend(cross_refs)

        # Analyze cross-references
        total_files = len(self.content_cache)
        files_with_refs = len(set(ref.source_file for ref in all_cross_refs))

        ref_by_type = Counter(ref.reference_type for ref in all_cross_refs)
        ref_by_confidence = [
            (ref.source_file, ref.confidence_score)
            for ref in all_cross_refs
        ]

        # Validate existing references
        validation_results = self.validate_cross_references()

        report = {
            'summary': {
                'total_files_indexed': total_files,
                'files_with_cross_references': files_with_refs,
                'total_cross_references_found': len(all_cross_refs),
                'average_refs_per_file': len(all_cross_refs) / files_with_refs if files_with_refs > 0 else 0,
                'high_confidence_refs': len([r for r in all_cross_refs if r.confidence_score >= 0.7])
            },
            'cross_references_by_type': dict(ref_by_type),
            'top_files_by_references': sorted(
                Counter(ref.source_file for ref in all_cross_refs).items(),
                key=lambda x: x[1], reverse=True
            )[:10],
            'validation_results': validation_results,
            'recommendations': self._generate_recommendations(all_cross_refs, validation_results)
        }

        return report

    def _generate_recommendations(self, cross_refs: List[CrossReference],
                                validation_results: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Check for broken links
        if validation_results['broken_links']:
            recommendations.append(
                f"Fix {len(validation_results['broken_links'])} broken links found in the documentation"
            )

        # Check coverage
        if len(cross_refs) < 100:
            recommendations.append(
                "Consider adding more cross-references to improve content connectivity"
            )

        # Check high-confidence uninserted references
        high_confidence_uninserted = len([r for r in cross_refs if r.confidence_score >= 0.8])
        if high_confidence_uninserted > 10:
            recommendations.append(
                f"Consider auto-inserting {high_confidence_uninserted} high-confidence cross-references"
            )

        # Check for under-referenced sections
        section_counts = Counter(
            self.knowledge_graph.concepts[ref.target_concept].section
            for ref in cross_refs
            if ref.target_concept in [c.name for c in self.knowledge_graph.concepts.values()]
        )

        if section_counts:
            min_refs = min(section_counts.values())
            max_refs = max(section_counts.values())
            if max_refs / min_refs > 5:
                recommendations.append(
                    "Some sections have significantly fewer cross-references - consider balancing the coverage"
                )

        return recommendations

    def batch_process_files(self, auto_insert: bool = False, max_files: int = None) -> Dict:
        """Process multiple files for cross-referencing"""
        # Index all files
        self.index_all_files()

        files_to_process = list(self.content_cache.keys())
        if max_files:
            files_to_process = files_to_process[:max_files]

        results = {
            'processed_files': 0,
            'total_suggestions': 0,
            'total_insertions': 0,
            'file_results': []
        }

        for file_path in files_to_process:
            try:
                file_result = self.insert_cross_references(file_path, auto_insert)
                results['file_results'].append(file_result)
                results['processed_files'] += 1
                results['total_suggestions'] += file_result['total_suggestions']
                results['total_insertions'] += file_result['inserted_count']

                logger.info(f"Processed {file_path}: {file_result['inserted_count']} insertions")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        # Save cache
        self._save_cache()

        return results

def main():
    """Main function to run the cross-referencing system"""
    # Initialize the system
    cross_ref_system = IntelligentCrossReferencer()

    # Process files
    results = cross_ref_system.batch_process_files(auto_insert=False, max_files=10)

    print("Cross-Reference Processing Results:")
    print(f"Processed files: {results['processed_files']}")
    print(f"Total suggestions: {results['total_suggestions']}")
    print(f"Total insertions: {results['total_insertions']}")

    # Generate report
    report = cross_ref_system.generate_cross_reference_report()
    print("\nCross-Reference Report:")
    print(json.dumps(report, indent=2, default=str))

    # Save report
    with open('/Users/dtumkorkmaz/Projects/ai-docs/cross_reference_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\nDetailed report saved to cross_reference_report.json")

if __name__ == "__main__":
    main()