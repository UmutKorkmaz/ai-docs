#!/usr/bin/env python3
"""
Integrates cross-reference links into markdown documentation files
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class LinkInfo:
    """Information about a cross-reference link"""
    target_id: str
    target_name: str
    target_section: str
    relationship_type: str
    strength: float
    direction: str

class MarkdownCrossReferenceIntegrator:
    """Integrates cross-reference links into markdown files"""

    def __init__(self, docs_root: str, cross_ref_data: str):
        self.docs_root = Path(docs_root)
        self.data_root = Path(cross_ref_data)

        # Load cross-reference data
        with open(self.data_root / "concepts.json") as f:
            self.concepts = json.load(f)

        with open(self.data_root / "cross_reference_links.json") as f:
            self.links = json.load(f)

        # Create ID to concept mapping
        self.id_to_concept = {c["id"]: c for c in self.concepts.values()}

    def integrate_all_files(self) -> None:
        """Integrate cross-references into all markdown files"""
        print("Integrating cross-references into markdown files...")

        for md_file in self.docs_root.rglob("*.md"):
            # Skip certain files
            if any(skip in str(md_file) for skip in [
                "node_modules", ".git", "cross_reference_output"
            ]):
                continue

            self.integrate_file(md_file)

        print("Integration complete!")

    def integrate_file(self, file_path: Path) -> None:
        """Integrate cross-references into a single file"""
        try:
            content = file_path.read_text(encoding='utf-8')

            # Check if file already has cross-references
            if "<!-- AI_DOCS_CROSS_REFERENCES -->" in content:
                return

            # Find all concepts mentioned in file
            mentioned_concepts = self._find_mentioned_concepts(content)

            if not mentioned_concepts:
                return

            # Generate cross-reference section
            cross_ref_section = self._generate_cross_reference_section(
                mentioned_concepts, file_path
            )

            # Add to end of file
            updated_content = content + "\n\n" + cross_ref_section

            # Write back
            file_path.write_text(updated_content, encoding='utf-8')

            print(f"Added cross-references to {file_path.relative_to(self.docs_root)}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def _find_mentioned_concepts(self, content: str) -> List[str]:
        """Find concepts mentioned in content"""
        mentioned = []

        # Look for exact matches of concept names
        for concept_id, concept in self.concepts.items():
            name = concept["name"]
            # Escape special regex characters
            pattern = re.escape(name)
            if re.search(pattern, content, re.IGNORECASE):
                mentioned.append(concept_id)

        return mentioned

    def _generate_cross_reference_section(self, concept_ids: List[str], file_path: Path) -> str:
        """Generate cross-reference section for markdown file"""
        section_lines = [
            "<!-- AI_DOCS_CROSS_REFERENCES -->",
            "",
            "## Cross-References",
            "",
            "*Automatically generated links to related concepts*",
            ""
        ]

        # Group links by type
        links_by_type = {
            "Prerequisites": [],
            "Applications": [],
            "Related Concepts": [],
            "Extensions": [],
            "Similar Implementations": []
        }

        for concept_id in concept_ids:
            if concept_id in self.links:
                for link in self.links[concept_id]:
                    # Skip if direction is incoming (already covered)
                    if link["direction"] == "incoming":
                        continue

                    # Categorize by relationship type
                    if "PREREQUISITE" in link["relationship_type"]:
                        links_by_type["Prerequisites"].append(link)
                    elif "APPLICATION" in link["relationship_type"]:
                        links_by_type["Applications"].append(link)
                    elif "EXTEND" in link["relationship_type"]:
                        links_by_type["Extensions"].append(link)
                    elif "SIMILAR" in link["relationship_type"] or "IMPLEMENTATION" in link["relationship_type"]:
                        links_by_type["Similar Implementations"].append(link)
                    else:
                        links_by_type["Related Concepts"].append(link)

        # Generate links for each category
        for category, link_list in links_by_type.items():
            if link_list:
                section_lines.append(f"### {category}")
                section_lines.append("")

                # Sort by strength
                link_list.sort(key=lambda x: x["relationship_strength"], reverse=True)

                for link in link_list[:5]:  # Top 5 per category
                    target = self.id_to_concept.get(link["concept_id"])
                    if target:
                        # Create relative path to target
                        target_path = self._create_relative_link(
                            file_path, target["file_path"], link["concept_id"]
                        )

                        section_lines.append(
                            f"- [{target['name']}]({target_path}) "
                            f"({link['relationship_type'].replace('_', ' ').lower()}, "
                            f"strength: {link['relationship_strength']:.2f})"
                        )

                section_lines.append("")

        # Add learning path suggestions
        section_lines.extend(self._generate_learning_path_suggestions(concept_ids))

        return "\n".join(section_lines)

    def _create_relative_link(self, source_file: Path, target_file: str, target_id: str) -> str:
        """Create relative link from source to target"""
        # For now, just create anchor links within the same page
        # In a real implementation, you'd calculate the relative path

        # Clean target file path
        target_path = Path(target_file).relative_to(self.docs_root)

        # Create anchor from concept ID
        anchor = target_id.lower().replace("_", "-")

        return f"{target_path}#{anchor}"

    def _generate_learning_path_suggestions(self, concept_ids: List[str]) -> List[str]:
        """Generate learning path suggestions"""
        suggestions = [
            "### Learning Path Suggestions",
            "",
            "*Based on current topic, consider exploring:*",
            ""
        ]

        # Simple path suggestions based on difficulty
        beginner_concepts = []
        advanced_concepts = []

        for concept_id in concept_ids:
            concept = self.id_to_concept[concept_id]
            if concept["difficulty"] in ["beginner", "intermediate"]:
                beginner_concepts.append(concept)
            else:
                advanced_concepts.append(concept)

        if beginner_concepts:
            suggestions.append("**Next Steps:**")
            for link in self.links.get(concept_ids[0], [])[:3]:
                if link["direction"] == "outgoing":
                    target = self.id_to_concept.get(link["concept_id"])
                    if target and target["difficulty"] in ["intermediate", "advanced"]:
                        suggestions.append(
                            f"- {target['name']} ({target['difficulty']})"
                        )
            suggestions.append("")

        if advanced_concepts:
            suggestions.append("**Foundation Concepts:**")
            for link in self.links.get(concept_ids[0], [])[:3]:
                if link["direction"] == "incoming":
                    target = self.id_to_concept.get(link["concept_id"])
                    if target and target["difficulty"] in ["beginner", "intermediate"]:
                        suggestions.append(
                            f"- {target['name']} ({target['difficulty']})"
                        )
            suggestions.append("")

        return suggestions

    def create_index_file(self) -> None:
        """Create a master index file with all concepts"""
        index_path = self.docs_root / "CROSS_REFERENCE_INDEX.md"

        index_content = [
            "# AI Documentation Cross-Reference Index",
            "",
            "*Generated automatically - do not edit manually*",
            "",
            "This index provides an overview of all concepts and their relationships in the AI documentation.",
            "",
            "## Quick Navigation",
            "",
            "- [By Section](#by-section)",
            "- [By Difficulty](#by-difficulty)",
            "- [By Category](#by-category)",
            "- [Relationship Map](#relationship-map)",
            "",
            "## By Section",
            ""
        ]

        # Group by section
        sections = {}
        for concept in self.concepts.values():
            section = concept["section"]
            if section not in sections:
                sections[section] = []
            sections[section].append(concept)

        # List concepts by section
        for section, concepts in sorted(sections.items()):
            index_content.append(f"### {section}")
            index_content.append("")

            for concept in sorted(concepts, key=lambda x: x["name"]):
                file_path = Path(concept["file_path"]).relative_to(self.docs_root)
                index_content.append(f"- [{concept['name']}]({file_path})")

            index_content.append("")

        # Add statistics
        index_content.extend([
            "## Statistics",
            "",
            f"- Total Concepts: {len(self.concepts)}",
            f"- Total Relationships: {sum(len(links) for links in self.links.values())}",
            f"- Sections: {len(sections)}",
            "",
            "## Relationship Map",
            "",
            "To explore the interactive knowledge graph, open [`knowledge_navigator.html`](components/knowledge_navigator.html) in your browser.",
            "",
            "## Search",
            "",
            "Use your browser's search function (Ctrl+F or Cmd+F) to find specific concepts.",
            ""
        ])

        # Write index file
        index_path.write_text("\n".join(index_content), encoding='utf-8')
        print(f"Created cross-reference index at {index_path}")

def main():
    """Main function"""
    docs_root = "/Users/dtumkorkmaz/Projects/ai-docs"
    cross_ref_data = "/Users/dtumkorkmaz/Projects/ai-docs/cross_reference_output"

    # Check if cross-reference data exists
    if not os.path.exists(cross_ref_data):
        print("Cross-reference data not found. Run cross_reference_generator.py first.")
        return

    # Initialize integrator
    integrator = MarkdownCrossReferenceIntegrator(docs_root, cross_ref_data)

    # Integrate all files
    integrator.integrate_all_files()

    # Create index
    integrator.create_index_file()

if __name__ == "__main__":
    main()