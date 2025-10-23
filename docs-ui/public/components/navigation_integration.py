#!/usr/bin/env python3
"""
AI Documentation Navigation Integration
=====================================

Main integration module that combines all navigation components and provides
a unified interface for the AI documentation navigation system.

Author: AI Documentation Team
Version: 1.0.0
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from navigation_system import AIDocumentationNavigator, NavigationState
from navigation_ui import NavigationUIRenderer, UIState


class AIDocumentationNavigationApp:
    """Main application class for the AI documentation navigation system."""

    def __init__(self, base_path: str = "/Users/dtumkorkmaz/Projects/ai-docs"):
        self.base_path = Path(base_path)
        self.config = self._load_config()

        # Initialize core components
        self.navigator = AIDocumentationNavigator(str(base_path))
        self.ui_renderer = NavigationUIRenderer(self.navigator)

        # Initialize state
        self.navigation_state = NavigationState()
        self.ui_state = UIState()
        self.user_data = self._load_user_data()

        # Ensure required directories exist
        self._ensure_directories()

    def _load_config(self) -> Dict:
        """Load navigation configuration."""
        config_path = self.base_path / "components" / "navigation_config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "version": "1.0.0",
            "ui_settings": {
                "default_theme": "auto",
                "default_font_size": 16,
                "sidebar_width": 320
            },
            "navigation_features": {
                "breadcrumbs": {"enabled": True},
                "search": {"enabled": True},
                "progress_tracking": {"enabled": True},
                "favorites": {"enabled": True},
                "keyboard_shortcuts": {"enabled": True}
            }
        }

    def _load_user_data(self) -> Dict:
        """Load user-specific data."""
        user_data_path = self.base_path / "components" / "user_data.json"
        try:
            with open(user_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "user_id": "default_user",
                "created_at": datetime.now().isoformat(),
                "completed_sections": [],
                "favorite_sections": [],
                "recent_sections": [],
                "learning_progress": {},
                "preferences": {}
            }

    def _save_user_data(self):
        """Save user-specific data."""
        user_data_path = self.base_path / "components" / "user_data.json"
        try:
            with open(user_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving user data: {e}")

    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            self.base_path / "components",
            self.base_path / "components" / "css",
            self.base_path / "components" / "js",
            self.base_path / "components" / "templates"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def generate_complete_html_page(self, section_id: Optional[str] = None) -> str:
        """Generate a complete HTML page with navigation."""
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Documentation - Navigation</title>
            <link rel="stylesheet" href="components/css/navigation.css">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        </head>
        <body>
            <!-- Navigation Sidebar -->
            {self.ui_renderer.render_sidebar(section_id)}

            <!-- Main Content Area -->
            <main class="main-content" id="main-content">
                <!-- Breadcrumbs -->
                {self.ui_renderer.render_breadcrumbs(section_id) if section_id else ''}

                <!-- Content will be loaded here -->
                <div class="content-area" id="content-area">
                    {self._get_content_for_section(section_id) if section_id else self._get_home_content()}
                </div>
            </main>

            <!-- Modals and Overlays -->
            {self.ui_renderer.render_search_interface()}
            {self.ui_renderer.render_quick_jump_modal()}
            {self.ui_renderer.render_keyboard_shortcuts_help()}
            {self.ui_renderer.render_favorites_panel(self.user_data.get('favorite_sections', []))}
            {self.ui_renderer.render_recent_sections_panel(self.user_data.get('recent_sections', []))}

            <!-- JavaScript -->
            <script src="components/js/navigation.js"></script>
            <script>
                // Initialize navigation system
                document.addEventListener('DOMContentLoaded', function() {{
                    initializeNavigation({json.dumps(self.user_data)});
                }});
            </script>
        </body>
        </html>
        """
        return html

    def _get_content_for_section(self, section_id: str) -> str:
        """Get content for a specific section."""
        if section_id not in self.navigator.sections:
            return self._get_not_found_content()

        section = self.navigator.sections[section_id]
        nav_data = self.navigator.get_section_navigation(section_id)

        return f"""
        <div class="section-content" data-section-id="{section_id}">
            <header class="section-header">
                <h1>{section.title}</h1>
                <p class="section-description">{section.description}</p>

                <div class="section-metadata">
                    <div class="metadata-item">
                        <i class="fas fa-layer-group"></i>
                        <span>Level {section.level} - {self._get_level_name(section.level)}</span>
                    </div>
                    <div class="metadata-item">
                        <i class="fas fa-clock"></i>
                        <span>Est. {section.estimated_time // 60}h {section.estimated_time % 60}m</span>
                    </div>
                    <div class="metadata-item">
                        <i class="fas fa-book"></i>
                        <span>{section.topics_count} topics</span>
                    </div>
                    {f'''<div class="metadata-item">
                        <i class="fas fa-laptop-code"></i>
                        <span>{section.interactive_notebooks} notebooks</span>
                    </div>''' if section.interactive_notebooks > 0 else ''}
                </div>

                <div class="section-actions">
                    <button class="btn btn-primary" onclick="markAsCompleted('{section_id}')">
                        <i class="fas fa-check"></i> Mark as Completed
                    </button>
                    <button class="btn btn-secondary" onclick="toggleFavorite('{section_id}')">
                        <i class="fas fa-star"></i> Add to Favorites
                    </button>
                    <button class="btn btn-outline" onclick="shareSection('{section_id}')">
                        <i class="fas fa-share"></i> Share
                    </button>
                </div>
            </header>

            <!-- Prerequisites -->
            {self._render_prerequisites(nav_data.get('prerequisites', []))}

            <!-- Quick Actions -->
            <div class="quick-actions-section">
                <h3>Quick Actions</h3>
                <div class="action-grid">
                    {self._render_quick_actions(nav_data.get('quick_actions', []))}
                </div>
            </div>

            <!-- Interactive Notebooks -->
            {self._render_interactive_notebooks(section) if section.interactive_notebooks > 0 else ''}

            <!-- Related Sections -->
            {self._render_related_sections(nav_data.get('related_sections', []))}

            <!-- See Also -->
            {self._render_see_also(nav_data.get('see_also', []))}

            <!-- Knowledge Graph -->
            {self._render_knowledge_graph(nav_data.get('knowledge_graph', []))}
        </div>
        """

    def _get_home_content(self) -> str:
        """Get home page content."""
        return f"""
        <div class="home-content">
            <header class="home-header">
                <h1>AI Documentation System</h1>
                <p>Comprehensive guide covering 25 major AI sections with 1500+ topics</p>
            </header>

            <!-- Learning Paths -->
            <section class="learning-paths-overview">
                <h2>🎓 Learning Paths</h2>
                <div class="paths-grid">
                    {self._render_learning_paths_overview()}
                </div>
            </section>

            <!-- Progress Overview -->
            <section class="progress-overview">
                <h2>📊 Your Progress</h2>
                {self.ui_renderer.render_progress_tracker(self.user_data)}
            </section>

            <!-- Quick Access -->
            <section class="quick-access">
                <h2>⚡ Quick Access</h2>
                <div class="access-grid">
                    {self._render_quick_access_sections()}
                </div>
            </section>

            <!-- Recent Updates -->
            <section class="recent-updates">
                <h2>🆕 Recent Updates</h2>
                {self._render_recent_updates()}
            </section>
        </div>
        """

    def _get_not_found_content(self) -> str:
        """Get 404 not found content."""
        return """
        <div class="not-found-content">
            <h1>Section Not Found</h1>
            <p>The requested section could not be found. Please check the URL and try again.</p>
            <button class="btn btn-primary" onclick="window.location.href='00_Overview.md'">
                <i class="fas fa-home"></i> Go to Home
            </button>
        </div>
        """

    def _get_level_name(self, level: int) -> str:
        """Get human-readable difficulty level name."""
        level_names = {
            1: "Beginner",
            2: "Intermediate",
            3: "Advanced",
            4: "Research/Expert"
        }
        return level_names.get(level, "Unknown")

    def _render_prerequisites(self, prerequisites: List[Dict]) -> str:
        """Render prerequisites section."""
        if not prerequisites:
            return ""

        return f"""
        <section class="prerequisites-section">
            <h3>📚 Prerequisites</h3>
            <div class="prerequisites-grid">
                {"".join([
                    f'''<div class="prerequisite-card">
                        <h4><a href="{prereq['path']}">{prereq['title']}</a></h4>
                        <p>{prereq['description']}</p>
                        <span class="prerequisite-level level-{prereq['level']}">
                            {self._get_level_name(prereq['level'])}
                        </span>
                    </div>'''
                    for prereq in prerequisites
                ])}
            </div>
        </section>
        """

    def _render_quick_actions(self, actions: List[Dict]) -> str:
        """Render quick actions section."""
        if not actions:
            return ""

        return "".join([
            f'''<button class="action-btn" onclick="{action['action']}">
                <i class="fas fa-{action['icon']}"></i>
                <span>{action['label']}</span>
            </button>'''
            for action in actions
        ])

    def _render_interactive_notebooks(self, section) -> str:
        """Render interactive notebooks section."""
        return f"""
        <section class="interactive-notebooks-section">
            <h3>📓 Interactive Notebooks ({section.interactive_notebooks} available)</h3>
            <p>Get hands-on experience with interactive Jupyter notebooks covering this section's topics.</p>
            <button class="btn btn-primary" onclick="openNotebooks('{section.id}')">
                <i class="fas fa-laptop-code"></i> Open Interactive Notebooks
            </button>
        </section>
        """

    def _render_related_sections(self, related_sections: List[Dict]) -> str:
        """Render related sections."""
        if not related_sections:
            return ""

        # Limit to top 5 related sections
        related_sections = related_sections[:5]

        return f"""
        <section class="related-sections">
            <h3>🔗 Related Sections</h3>
            <div class="related-grid">
                {"".join([
                    f'''<div class="related-card">
                        <h4><a href="{section['path']}">{section['title']}</a></h4>
                        <p>{section['description'][:100]}...</p>
                        <div class="related-meta">
                            <span class="difficulty level-{section['level']}">
                                {self._get_level_name(section['level'])}
                            </span>
                            <span class="time">🕐 {section['estimated_time'] // 60}h</span>
                        </div>
                    </div>'''
                    for section in related_sections
                ])}
            </div>
        </section>
        """

    def _render_see_also(self, suggestions: List[Dict]) -> str:
        """Render "See Also" suggestions."""
        if not suggestions:
            return ""

        return f"""
        <section class="see-also-section">
            <h3>📖 See Also</h3>
            <div class="suggestions-list">
                {"".join([
                    f'''<div class="suggestion-item">
                        <span class="suggestion-type">{suggestion['type'].replace('_', ' ').title()}:</span>
                        <a href="{suggestion['path']}">{suggestion['title']}</a>
                        <p>{suggestion['description']}</p>
                    </div>'''
                    for suggestion in suggestions
                ])}
            </div>
        </section>
        """

    def _render_knowledge_graph(self, knowledge_graph: List[Dict]) -> str:
        """Render knowledge graph visualization."""
        if not knowledge_graph:
            return ""

        return f"""
        <section class="knowledge-graph-section">
            <h3>🕸️ Knowledge Graph</h3>
            <p>Explore the conceptual relationships between this section and related topics.</p>
            <div class="knowledge-graph-visualization" id="knowledge-graph">
                <!-- Interactive visualization would be rendered here -->
                <div class="graph-placeholder">
                    <i class="fas fa-project-diagram"></i>
                    <p>Interactive knowledge graph visualization</p>
                    <button class="btn btn-outline" onclick="expandKnowledgeGraph()">
                        <i class="fas fa-expand"></i> Expand Graph
                    </button>
                </div>
            </div>
        </section>
        """

    def _render_learning_paths_overview(self) -> str:
        """Render learning paths overview."""
        html = ""
        for path_id, path in self.navigator.learning_paths.items():
            html += f"""
            <div class="path-overview-card" onclick="navigateToPath('{path_id}')">
                <h3>{path.name}</h3>
                <p>{path.description}</p>
                <div class="path-stats">
                    <span class="stat"><i class="fas fa-clock"></i> {path.estimated_duration}h</span>
                    <span class="stat"><i class="fas fa-layer-group"></i> Level {path.difficulty_level}</span>
                    <span class="stat"><i class="fas fa-book"></i> {len(path.sections)} sections</span>
                </div>
                <button class="btn btn-primary">Start Learning Path</button>
            </div>
            """
        return html

    def _render_quick_access_sections(self) -> str:
        """Render quick access sections."""
        popular_sections = [
            "01_foundational_ml", "02_advanced_dl", "03_nlp",
            "04_computer_vision", "05_generative_ai", "07_ai_ethics_safety"
        ]

        html = ""
        for section_id in popular_sections:
            if section_id in self.navigator.sections:
                section = self.navigator.sections[section_id]
                html += f"""
                <div class="quick-access-card" onclick="navigateToSection('{section_id}')">
                    <h4>{section.title}</h4>
                    <p>{section.description[:80]}...</p>
                    <span class="difficulty level-{section.level}">
                        {self._get_level_name(section.level)}
                    </span>
                </div>
                """
        return html

    def _render_recent_updates(self) -> str:
        """Render recent updates section."""
        return """
        <div class="updates-list">
            <div class="update-item">
                <span class="update-date">2024-10-05</span>
                <h4>New Section: Emerging Research 2025</h4>
                <p>Latest AI research trends and emerging topics for 2025.</p>
            </div>
            <div class="update-item">
                <span class="update-date">2024-10-03</span>
                <h4>Updated: AI Policy and Regulation</h4>
                <p>Added new compliance frameworks and regulatory updates.</p>
            </div>
            <div class="update-item">
                <span class="update-date">2024-10-01</span>
                <h4>Enhanced: Generative AI Section</h4>
                <p>New content on state space models and multimodal generation.</p>
            </div>
        </div>
        """

    def generate_css_file(self) -> str:
        """Generate the main CSS file."""
        return self.ui_renderer.generate_css()

    def generate_js_file(self) -> str:
        """Generate the main JavaScript file."""
        return self.ui_renderer.generate_javascript()

    def generate_search_results_page(self, query: str, filters: Dict = None) -> str:
        """Generate search results page."""
        search_results = self.navigator.search_sections(query, filters)

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Search Results: {query} - AI Documentation</title>
            <link rel="stylesheet" href="components/css/navigation.css">
        </head>
        <body>
            {self.ui_renderer.render_sidebar()}

            <main class="main-content">
                {self.ui_renderer.render_breadcrumbs(None)}

                <div class="search-results-container">
                    <header class="search-header">
                        <h1>Search Results</h1>
                        <p class="search-query">Showing {len(search_results)} results for "{query}"</p>
                    </header>

                    <div class="search-filters-applied">
                        {self._render_applied_filters(filters) if filters else ''}
                    </div>

                    <div class="search-results-list">
                        {self._render_search_results(search_results) if search_results else self._render_no_results(query)}
                    </div>
                </div>
            </main>

            <script src="components/js/navigation.js"></script>
        </body>
        </html>
        """
        return html

    def _render_applied_filters(self, filters: Dict) -> str:
        """Render applied search filters."""
        if not filters:
            return ""

        html = '<div class="applied-filters"><span>Filters:</span>'
        for key, value in filters.items():
            if value:
                html += f'<span class="filter-tag">{key}: {value}</span>'
        html += '</div>'
        return html

    def _render_search_results(self, results: List[Dict]) -> str:
        """Render search results."""
        html = ""
        for result in results:
            section = result['section']
            html += f"""
            <div class="search-result-item">
                <h3><a href="{section['path']}">{section['title']}</a></h3>
                <p>{section['description']}</p>
                <div class="result-meta">
                    <span class="difficulty level-{section['level']}">
                        {self._get_level_name(section['level'])}
                    </span>
                    <span class="category">{section['category'].replace('_', ' ').title()}</span>
                    <span class="relevance">Relevance: {result['relevance_score']}</span>
                </div>
                <div class="match-reasons">
                    {", ".join(result['match_reasons'])}
                </div>
            </div>
            """
        return html

    def _render_no_results(self, query: str) -> str:
        """Render no results message."""
        return f"""
        <div class="no-results">
            <h3>No results found for "{query}"</h3>
            <p>Try:</p>
            <ul>
                <li>Using different keywords</li>
                <li>Checking your spelling</li>
                <li>Using more general terms</li>
                <li>Browsing the sections directly</li>
            </ul>
        </div>
        """

    def export_navigation_data(self, format: str = "json") -> str:
        """Export all navigation data."""
        return self.navigator.export_navigation_data(format)

    def update_user_progress(self, section_id: str, action: str) -> bool:
        """Update user progress for a section."""
        if section_id not in self.navigator.sections:
            return False

        if action == "complete":
            if section_id not in self.user_data["completed_sections"]:
                self.user_data["completed_sections"].append(section_id)
                # Add to recent sections
                self._add_to_recent(section_id)
        elif action == "favorite":
            if section_id not in self.user_data["favorite_sections"]:
                self.user_data["favorite_sections"].append(section_id)
        elif action == "unfavorite":
            if section_id in self.user_data["favorite_sections"]:
                self.user_data["favorite_sections"].remove(section_id)

        self._save_user_data()
        return True

    def _add_to_recent(self, section_id: str):
        """Add section to recent sections."""
        recent = self.user_data.get("recent_sections", [])
        recent = [s for s in recent if s != section_id]  # Remove if exists
        recent.insert(0, section_id)  # Add to beginning
        recent = recent[:10]  # Keep only 10 recent
        self.user_data["recent_sections"] = recent

    def get_user_statistics(self) -> Dict:
        """Get user learning statistics."""
        completed = len(self.user_data.get("completed_sections", []))
        favorites = len(self.user_data.get("favorite_sections", []))
        total_sections = len(self.navigator.sections)

        progress_percent = (completed / total_sections * 100) if total_sections > 0 else 0

        return {
            "completed_sections": completed,
            "total_sections": total_sections,
            "progress_percent": progress_percent,
            "favorite_sections": favorites,
            "recent_sections": len(self.user_data.get("recent_sections", []))
        }

    def build_static_files(self):
        """Build all static files for the navigation system."""
        # Create directories
        css_dir = self.base_path / "components" / "css"
        js_dir = self.base_path / "components" / "js"
        templates_dir = self.base_path / "components" / "templates"

        css_dir.mkdir(parents=True, exist_ok=True)
        js_dir.mkdir(parents=True, exist_ok=True)
        templates_dir.mkdir(parents=True, exist_ok=True)

        # Generate CSS file
        css_content = self.generate_css_file()
        with open(css_dir / "navigation.css", "w", encoding="utf-8") as f:
            f.write(css_content)

        # Generate JavaScript file
        js_content = self.generate_js_file()
        with open(js_dir / "navigation.js", "w", encoding="utf-8") as f:
            f.write(js_content)

        # Generate main navigation page
        main_html = self.generate_complete_html_page()
        with open(self.base_path / "navigation.html", "w", encoding="utf-8") as f:
            f.write(main_html)

        # Generate search page template
        search_html = self.generate_search_results_page("test query")
        with open(templates_dir / "search_results.html", "w", encoding="utf-8") as f:
            f.write(search_html)

        print(f"✅ Navigation system files built successfully!")
        print(f"   - CSS: {css_dir / 'navigation.css'}")
        print(f"   - JavaScript: {js_dir / 'navigation.js'}")
        print(f"   - Main page: {self.base_path / 'navigation.html'}")
        print(f"   - Search template: {templates_dir / 'search_results.html'}")


def main():
    """Main function to demonstrate the navigation system."""
    print("🚀 AI Documentation Navigation System")
    print("=" * 50)

    # Initialize the navigation app
    app = AIDocumentationNavigationApp()

    # Build static files
    app.build_static_files()

    # Print statistics
    stats = app.get_user_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   - Total sections: {stats['total_sections']}")
    print(f"   - Learning paths: {len(app.navigator.learning_paths)}")
    print(f"   - Interactive notebooks: {sum(s.interactive_notebooks for s in app.navigator.sections.values())}")

    print(f"\n🎯 Features Enabled:")
    features = app.config.get("navigation_features", {})
    for feature, config in features.items():
        if isinstance(config, dict) and config.get("enabled"):
            print(f"   ✅ {feature.replace('_', ' ').title()}")
        elif isinstance(config, bool) and config:
            print(f"   ✅ {feature.replace('_', ' ').title()}")

    print(f"\n🎨 Generated Files:")
    print(f"   - Complete HTML navigation page")
    print(f"   - Responsive CSS styles")
    print(f"   - Interactive JavaScript")
    print(f"   - Search functionality")
    print(f"   - Progress tracking")

    print(f"\n🔗 Quick Access:")
    print(f"   - Open: navigation.html")
    print(f"   - Config: components/navigation_config.json")
    print(f"   - User data: components/user_data.json")

    print(f"\n✨ Navigation system is ready to use!")


if __name__ == "__main__":
    main()