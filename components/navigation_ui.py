#!/usr/bin/env python3
"""
AI Documentation Navigation UI Components
=========================================

User interface components for the AI documentation navigation system.
Includes sidebar navigation, breadcrumbs, search interface, and progress tracking.

Author: AI Documentation Team
Version: 1.0.0
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class UIState:
    """Current UI state for the navigation system."""
    active_section: Optional[str] = None
    sidebar_open: bool = True
    search_open: bool = False
    current_path: Optional[str] = None
    view_mode: str = "document"  # document, grid, list
    theme: str = "light"  # light, dark, auto
    font_size: int = 16


class NavigationUIRenderer:
    """Renders UI components for the navigation system."""

    def __init__(self, navigation_system):
        self.nav_system = navigation_system
        self.ui_state = UIState()

    def render_sidebar(self, current_section: Optional[str] = None) -> str:
        """Render the sidebar navigation component."""
        html = f"""
        <div class="ai-docs-sidebar" id="sidebar">
            <div class="sidebar-header">
                <h2>AI Documentation</h2>
                <button class="sidebar-toggle" onclick="toggleSidebar()">×</button>
            </div>

            <div class="sidebar-search">
                <input type="text"
                       id="nav-search"
                       placeholder="Search sections..."
                       onkeyup="handleSearch(event)"
                       autocomplete="off">
                <button onclick="performSearch()">🔍</button>
            </div>

            <div class="nav-sections">
                {self._render_sections_nav(current_section)}
            </div>

            <div class="learning-paths">
                <h3>Learning Paths</h3>
                {self._render_learning_paths_nav()}
            </div>

            <div class="quick-actions">
                <h3>Quick Actions</h3>
                <button onclick="showKeyboardShortcuts()">⌨️ Shortcuts</button>
                <button onclick="showProgress()">📊 Progress</button>
                <button onclick="showFavorites()">⭐ Favorites</button>
                <button onclick="showRecent()">🕐 Recent</button>
            </div>
        </div>
        """
        return html

    def _render_sections_nav(self, current_section: Optional[str]) -> str:
        """Render the sections navigation tree."""
        html = '<div class="sections-tree">'

        # Group sections by category
        categories = {}
        for section_id, section in self.nav_system.sections.items():
            if section.category not in categories:
                categories[section.category] = []
            categories[section.category].append(section)

        # Category display names
        category_names = {
            "foundations": "🎯 Foundations",
            "core": "⚡ Core Technologies",
            "advanced": "🚀 Advanced Topics",
            "applications": "🏢 Applications",
            "research": "🔬 Research",
            "business": "💼 Business & Enterprise",
            "technical": "⚙️ Technical & Infrastructure",
            "security": "🛡️ Security & Defense",
            "social": "🤝 Social & Ethical",
            "policy": "📋 Policy & Regulation",
            "human": "👥 Human-AI Interaction",
            "creative": "🎨 Creative & Entertainment",
            "industry": "🏭 Industry Applications",
            "legal": "⚖️ Legal & Regulatory"
        }

        for category, sections in sorted(categories.items()):
            html += f"""
            <div class="nav-category">
                <h4 onclick="toggleCategory('{category}')" class="category-toggle">
                    {category_names.get(category, category.title())}
                    <span class="category-count">({len(sections)})</span>
                </h4>
                <div class="category-sections" id="category-{category}">
            """

            for section in sorted(sections, key=lambda s: s.id):
                is_active = section.id == current_section
                is_completed = section.id in getattr(self.ui_state, 'completed_sections', set())
                has_notebooks = section.interactive_notebooks > 0

                html += f"""
                <div class="nav-section {('active' if is_active else '')} {('completed' if is_completed else '')}"
                     onclick="navigateToSection('{section.id}')">
                    <div class="section-info">
                        <span class="section-title">{section.title}</span>
                        <div class="section-meta">
                            <span class="difficulty level-{section.level}">
                                {'Beginner' if section.level == 1 else
                                 'Intermediate' if section.level == 2 else
                                 'Advanced' if section.level == 3 else 'Research'}
                            </span>
                            {f'<span class="notebooks">📓 {section.interactive_notebooks}</span>' if has_notebooks else ''}
                            {f'<span class="completed">✓</span>' if is_completed else ''}
                        </div>
                    </div>
                    <div class="section-time">
                        {section.estimated_time // 60}h {section.estimated_time % 60}m
                    </div>
                </div>
                """

            html += '</div></div>'

        html += '</div>'
        return html

    def _render_learning_paths_nav(self) -> str:
        """Render learning paths navigation."""
        html = '<div class="paths-nav">'

        for path_id, path in self.nav_system.learning_paths.items():
            html += f"""
            <div class="path-card" onclick="navigateToPath('{path_id}')">
                <h4>{path.name}</h4>
                <p>{path.description}</p>
                <div class="path-meta">
                    <span class="duration">🕐 {path.estimated_duration}h</span>
                    <span class="difficulty">📊 Level {path.difficulty_level}</span>
                    <span class="sections">📚 {len(path.sections)} sections</span>
                </div>
            </div>
            """

        html += '</div>'
        return html

    def render_breadcrumbs(self, current_section: str) -> str:
        """Render breadcrumb navigation."""
        nav_data = self.nav_system.get_section_navigation(current_section)
        breadcrumbs = nav_data.get('breadcrumb', [])

        html = '<nav class="breadcrumbs" aria-label="Breadcrumb navigation">'

        for i, crumb in enumerate(breadcrumbs):
            is_last = i == len(breadcrumbs) - 1
            html += f"""
            <a href="{crumb['path']}" class="breadcrumb-item {('active' if is_last else '')}">
                {crumb['title']}
            </a>
            """
            if not is_last:
                html += '<span class="breadcrumb-separator">›</span>'

        html += '</nav>'
        return html

    def render_progress_tracker(self, user_progress: Dict) -> str:
        """Render progress tracking component."""
        completed_sections = user_progress.get('completed_sections', set())
        total_sections = len(self.nav_system.sections)
        completed_count = len(completed_sections)
        progress_percent = (completed_count / total_sections) * 100 if total_sections > 0 else 0

        html = f"""
        <div class="progress-tracker">
            <div class="progress-header">
                <h3>📊 Your Learning Progress</h3>
                <span class="progress-percentage">{progress_percent:.1f}% Complete</span>
            </div>

            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_percent}%"></div>
            </div>

            <div class="progress-stats">
                <div class="stat">
                    <span class="stat-value">{completed_count}</span>
                    <span class="stat-label">Completed</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{total_sections - completed_count}</span>
                    <span class="stat-label">Remaining</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{len(user_progress.get('favorite_sections', []))}</span>
                    <span class="stat-label">Favorites</span>
                </div>
            </div>

            <div class="progress-details">
                <h4>Recently Completed</h4>
                <div class="recent-completed">
        """

        # Show recently completed sections
        recent_completed = user_progress.get('recently_completed', [])[:5]
        for section_id in recent_completed:
            if section_id in self.nav_system.sections:
                section = self.nav_system.sections[section_id]
                html += f"""
                <div class="completed-item">
                    <span class="completed-title">{section.title}</span>
                    <span class="completed-date">Completed today</span>
                </div>
                """

        html += """
                </div>
            </div>
        </div>
        """

        return html

    def render_search_interface(self) -> str:
        """Render the search interface."""
        html = """
        <div class="search-interface" id="search-modal" style="display: none;">
            <div class="search-header">
                <h3>🔍 Search AI Documentation</h3>
                <button class="close-search" onclick="closeSearch()">×</button>
            </div>

            <div class="search-form">
                <input type="text"
                       id="search-input"
                       placeholder="Search for topics, techniques, or applications..."
                       onkeyup="handleSearchInput(event)"
                       autofocus>

                <div class="search-filters">
                    <div class="filter-group">
                        <label>Difficulty Level:</label>
                        <select id="difficulty-filter" onchange="updateSearch()">
                            <option value="">All Levels</option>
                            <option value="1">Beginner</option>
                            <option value="2">Intermediate</option>
                            <option value="3">Advanced</option>
                            <option value="4">Research</option>
                        </select>
                    </div>

                    <div class="filter-group">
                        <label>Category:</label>
                        <select id="category-filter" onchange="updateSearch()">
                            <option value="">All Categories</option>
                            <option value="foundations">Foundations</option>
                            <option value="core">Core Technologies</option>
                            <option value="applications">Applications</option>
                            <option value="research">Research</option>
                            <option value="business">Business & Enterprise</option>
                        </select>
                    </div>

                    <div class="filter-group">
                        <label>Has Notebooks:</label>
                        <input type="checkbox" id="notebooks-filter" onchange="updateSearch()">
                    </div>
                </div>

                <button class="search-button" onclick="performSearch()">Search</button>
            </div>

            <div class="search-results" id="search-results">
                <div class="search-placeholder">
                    <p>Enter a search term to find relevant sections.</p>
                    <p>Try searching for: "ethics", "transformers", "computer vision", "reinforcement learning"</p>
                </div>
            </div>
        </div>
        """

        return html

    def render_quick_jump_modal(self) -> str:
        """Render the quick jump modal."""
        quick_jump = self.nav_system.get_quick_jump_options()

        html = """
        <div class="quick-jump-modal" id="quick-jump-modal" style="display: none;">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>⚡ Quick Jump</h3>
                    <button class="modal-close" onclick="closeQuickJump()">×</button>
                </div>

                <div class="quick-jump-tabs">
                    <button class="tab-button active" onclick="showJumpTab('categories')">Categories</button>
                    <button class="tab-button" onclick="showJumpTab('difficulty')">Difficulty</button>
                    <button class="tab-button" onclick="showJumpTab('popular')">Popular</button>
                </div>

                <div class="quick-jump-content">
                    <div id="jump-categories" class="jump-tab-content active">
        """

        for category, sections in quick_jump["categories"].items():
            html += f"""
            <div class="jump-category">
                <h4>{category}</h4>
                <div class="jump-sections">
            """
            for section_id in sections:
                if section_id in self.nav_system.sections:
                    section = self.nav_system.sections[section_id]
                    html += f"""
                    <button class="jump-section" onclick="quickJumpTo('{section_id}')">
                        {section.title}
                    </button>
                    """
            html += "</div></div>"

        html += """
                    </div>

                    <div id="jump-difficulty" class="jump-tab-content">
        """

        for level, sections in quick_jump["difficulty_levels"].items():
            html += f"""
            <div class="jump-difficulty">
                <h4>{level}</h4>
                <div class="jump-sections">
            """
            for section in sections:
                html += f"""
                <button class="jump-section" onclick="quickJumpTo('{section.id}')">
                    {section.title}
                </button>
                """
            html += "</div></div>"

        html += """
                    </div>

                    <div id="jump-popular" class="jump-tab-content">
        """

        for section_id in quick_jump["most_accessed"]:
            if section_id in self.nav_system.sections:
                section = self.nav_system.sections[section_id]
                html += f"""
                <button class="jump-section popular" onclick="quickJumpTo('{section_id}')">
                    {section.title}
                </button>
                """

        html += """
                    </div>
                </div>
            </div>
        </div>
        """

        return html

    def render_keyboard_shortcuts_help(self) -> str:
        """Render keyboard shortcuts help modal."""
        shortcuts = self.nav_system.get_keyboard_shortcuts()

        html = """
        <div class="shortcuts-modal" id="shortcuts-modal" style="display: none;">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>⌨️ Keyboard Shortcuts</h3>
                    <button class="modal-close" onclick="closeShortcuts()">×</button>
                </div>

                <div class="shortcuts-content">
        """

        for category, category_shortcuts in shortcuts.items():
            html += f"""
            <div class="shortcut-category">
                <h4>{category.title()}</h4>
                <div class="shortcuts-list">
            """

            for key, description in category_shortcuts.items():
                html += f"""
                <div class="shortcut-item">
                    <kbd class="shortcut-key">{key}</kbd>
                    <span class="shortcut-description">{description}</span>
                </div>
                """

            html += "</div></div>"

        html += """
                </div>
            </div>
        </div>
        """

        return html

    def render_favorites_panel(self, favorites: List[str]) -> str:
        """Render the favorites panel."""
        html = """
        <div class="favorites-panel" id="favorites-panel" style="display: none;">
            <div class="panel-header">
                <h3>⭐ Favorite Sections</h3>
                <button class="panel-close" onclick="closeFavorites()">×</button>
            </div>

            <div class="favorites-content">
        """

        if not favorites:
            html += '<p class="no-favorites">No favorite sections yet. Click the star icon on any section to add it here.</p>'
        else:
            for section_id in favorites:
                if section_id in self.nav_system.sections:
                    section = self.nav_system.sections[section_id]
                    nav_data = self.nav_system.get_section_navigation(section_id)

                    html += f"""
                    <div class="favorite-item">
                        <h4><a href="{section.path}">{section.title}</a></h4>
                        <p>{section.description}</p>
                        <div class="favorite-meta">
                            <span class="difficulty level-{section.level}">
                                {'Beginner' if section.level == 1 else
                                 'Intermediate' if section.level == 2 else
                                 'Advanced' if section.level == 3 else 'Research'}
                            </span>
                            <span class="time">🕐 {section.estimated_time // 60}h {section.estimated_time % 60}m</span>
                            <button class="remove-favorite" onclick="removeFavorite('{section_id}')">Remove</button>
                        </div>
                    </div>
                    """

        html += """
            </div>
        </div>
        """

        return html

    def render_recent_sections_panel(self, recent_sections: List[str]) -> str:
        """Render the recent sections panel."""
        html = """
        <div class="recent-panel" id="recent-panel" style="display: none;">
            <div class="panel-header">
                <h3>🕐 Recently Viewed</h3>
                <button class="panel-close" onclick="closeRecent()">×</button>
            </div>

            <div class="recent-content">
        """

        if not recent_sections:
            html += '<p class="no-recent">No recently viewed sections.</p>'
        else:
            for section_id in recent_sections:
                if section_id in self.nav_system.sections:
                    section = self.nav_system.sections[section_id]

                    html += f"""
                    <div class="recent-item">
                        <h4><a href="{section.path}">{section.title}</a></h4>
                        <p>{section.description[:100]}...</p>
                        <div class="recent-meta">
                            <span class="difficulty level-{section.level}">
                                {'Beginner' if section.level == 1 else
                                 'Intermediate' if section.level == 2 else
                                 'Advanced' if section.level == 3 else 'Research'}
                            </span>
                            <span class="time">Viewed recently</span>
                        </div>
                    </div>
                    """

        html += """
            </div>
        </div>
        """

        return html

    def generate_javascript(self) -> str:
        """Generate JavaScript for navigation interactions."""
        return f"""
        // AI Documentation Navigation JavaScript
        let navigationState = {{
            currentSection: null,
            completedSections: new Set({json.dumps(list(getattr(self.ui_state, 'completed_sections', [])))}),
            favoriteSections: new Set({json.dumps(list(getattr(self.ui_state, 'favorite_sections', [])))}),
            recentSections: [],
            sidebarOpen: true
        }};

        // Navigation functions
        function navigateToSection(sectionId) {{
            window.location.href = `{{{{{self.nav_system.base_path}}}/${{{{section_id}}}}}/00_Overview.md`;
        }}

        function navigateToPath(pathId) {{
            window.location.href = `{{{{{self.nav_system.base_path}}}}/learning_paths/${{{{pathId}}}}.html`;
        }}

        function quickJumpTo(sectionId) {{
            closeQuickJump();
            navigateToSection(sectionId);
        }}

        // UI control functions
        function toggleSidebar() {{
            const sidebar = document.getElementById('sidebar');
            navigationState.sidebarOpen = !navigationState.sidebarOpen;
            sidebar.classList.toggle('closed');
            localStorage.setItem('sidebarOpen', navigationState.sidebarOpen);
        }}

        function toggleCategory(categoryId) {{
            const category = document.getElementById(`category-${{categoryId}}`);
            category.classList.toggle('collapsed');
        }}

        function showQuickJump() {{
            document.getElementById('quick-jump-modal').style.display = 'block';
        }}

        function closeQuickJump() {{
            document.getElementById('quick-jump-modal').style.display = 'none';
        }}

        function showKeyboardShortcuts() {{
            document.getElementById('shortcuts-modal').style.display = 'block';
        }}

        function closeShortcuts() {{
            document.getElementById('shortcuts-modal').style.display = 'none';
        }}

        function showSearch() {{
            document.getElementById('search-modal').style.display = 'block';
            document.getElementById('search-input').focus();
        }}

        function closeSearch() {{
            document.getElementById('search-modal').style.display = 'none';
        }}

        function showFavorites() {{
            document.getElementById('favorites-panel').style.display = 'block';
        }}

        function closeFavorites() {{
            document.getElementById('favorites-panel').style.display = 'none';
        }}

        function showRecent() {{
            document.getElementById('recent-panel').style.display = 'block';
        }}

        function closeRecent() {{
            document.getElementById('recent-panel').style.display = 'none';
        }}

        // Search functions
        function handleSearch(event) {{
            if (event.key === 'Enter') {{
                performSearch();
            }}
        }}

        function performSearch() {{
            const query = document.getElementById('nav-search').value;
            if (query.trim()) {{
                window.location.href = `search.html?q=${{encodeURIComponent(query)}}`;
            }}
        }}

        function handleSearchInput(event) {{
            if (event.key === 'Enter') {{
                updateSearch();
            }}
        }}

        function updateSearch() {{
            const query = document.getElementById('search-input').value;
            const difficulty = document.getElementById('difficulty-filter').value;
            const category = document.getElementById('category-filter').value;
            const hasNotebooks = document.getElementById('notebooks-filter').checked;

            // This would typically make an AJAX call to get search results
            console.log('Searching:', {{ query, difficulty, category, hasNotebooks }});
        }}

        // Quick jump tab functions
        function showJumpTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.jump-tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab-button').forEach(button => {{
                button.classList.remove('active');
            }});

            // Show selected tab
            document.getElementById(`jump-${{tabName}}`).classList.add('active');
            event.target.classList.add('active');
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {{
            if (event.ctrlKey) {{
                switch(event.key) {{
                    case 'k':
                        event.preventDefault();
                        showSearch();
                        break;
                    case '/':
                        event.preventDefault();
                        showKeyboardShortcuts();
                        break;
                    case 'h':
                        event.preventDefault();
                        window.location.href = '00_Overview.md';
                        break;
                    case 'b':
                        event.preventDefault();
                        toggleSidebar();
                        break;
                    case 'p':
                        event.preventDefault();
                        // Show learning paths
                        break;
                }}
            }} else if (event.altKey) {{
                switch(event.key) {{
                    case 'ArrowLeft':
                        // Navigate to previous section
                        break;
                    case 'ArrowRight':
                        // Navigate to next section
                        break;
                }}
            }}
        }});

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            // Load saved preferences
            const sidebarOpen = localStorage.getItem('sidebarOpen');
            if (sidebarOpen !== null) {{
                navigationState.sidebarOpen = sidebarOpen === 'true';
                if (!navigationState.sidebarOpen) {{
                    document.getElementById('sidebar').classList.add('closed');
                }}
            }}

            // Track current section
            const currentPath = window.location.pathname;
            // Extract section ID from path and update navigation state
        }});

        // Utility functions
        function addFavorite(sectionId) {{
            navigationState.favoriteSections.add(sectionId);
            localStorage.setItem('favoriteSections', JSON.stringify([...navigationState.favoriteSections]));
            // Update UI
            location.reload();
        }}

        function removeFavorite(sectionId) {{
            navigationState.favoriteSections.delete(sectionId);
            localStorage.setItem('favoriteSections', JSON.stringify([...navigationState.favoriteSections]));
            // Update UI
            location.reload();
        }}

        function markCompleted(sectionId) {{
            navigationState.completedSections.add(sectionId);
            localStorage.setItem('completedSections', JSON.stringify([...navigationState.completedSections]));
            // Update UI
            location.reload();
        }}

        function addToRecent(sectionId) {{
            navigationState.recentSections = navigationState.recentSections.filter(id => id !== sectionId);
            navigationState.recentSections.unshift(sectionId);
            navigationState.recentSections = navigationState.recentSections.slice(0, 10); // Keep only 10 recent
            localStorage.setItem('recentSections', JSON.stringify(navigationState.recentSections));
        }}
        """

    def generate_css(self) -> str:
        """Generate CSS styles for the navigation system."""
        return """
        /* AI Documentation Navigation Styles */

        /* Sidebar Styles */
        .ai-docs-sidebar {
            position: fixed;
            left: 0;
            top: 0;
            width: 320px;
            height: 100vh;
            background: #f8f9fa;
            border-right: 1px solid #dee2e6;
            overflow-y: auto;
            z-index: 1000;
            transition: transform 0.3s ease;
        }

        .ai-docs-sidebar.closed {
            transform: translateX(-320px);
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .sidebar-header h2 {
            margin: 0;
            font-size: 1.2em;
            color: #495057;
        }

        .sidebar-toggle {
            background: none;
            border: none;
            font-size: 1.5em;
            cursor: pointer;
            padding: 5px;
            border-radius: 3px;
        }

        .sidebar-toggle:hover {
            background: #e9ecef;
        }

        /* Search Styles */
        .sidebar-search {
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            gap: 10px;
        }

        .sidebar-search input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }

        .sidebar-search button {
            padding: 8px 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        /* Navigation Sections */
        .nav-sections {
            padding: 15px 0;
        }

        .nav-category {
            margin-bottom: 15px;
        }

        .category-toggle {
            width: 100%;
            padding: 10px 15px;
            background: none;
            border: none;
            text-align: left;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            color: #495057;
        }

        .category-toggle:hover {
            background: #e9ecef;
        }

        .category-count {
            font-size: 0.8em;
            color: #6c757d;
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 10px;
        }

        .category-sections {
            padding-left: 15px;
        }

        .nav-section {
            padding: 12px 15px;
            cursor: pointer;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .nav-section:hover {
            background: #e9ecef;
        }

        .nav-section.active {
            background: #e3f2fd;
            border-left-color: #2196f3;
        }

        .nav-section.completed {
            border-left-color: #4caf50;
        }

        .section-info {
            flex: 1;
        }

        .section-title {
            display: block;
            font-weight: 500;
            margin-bottom: 4px;
        }

        .section-meta {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .difficulty {
            font-size: 0.75em;
            padding: 2px 6px;
            border-radius: 3px;
            background: #e9ecef;
        }

        .level-1 { background: #c8e6c9; color: #2e7d32; }
        .level-2 { background: #fff3cd; color: #856404; }
        .level-3 { background: #f8d7da; color: #721c24; }
        .level-4 { background: #e1bee7; color: #6a1b9a; }

        .notebooks {
            font-size: 0.8em;
            color: #2196f3;
        }

        .completed {
            color: #4caf50;
            font-weight: bold;
        }

        .section-time {
            font-size: 0.8em;
            color: #6c757d;
        }

        /* Learning Paths */
        .learning-paths {
            padding: 15px;
            border-top: 1px solid #dee2e6;
        }

        .learning-paths h3 {
            margin: 0 0 15px 0;
            font-size: 1em;
            color: #495057;
        }

        .path-card {
            padding: 12px;
            margin-bottom: 10px;
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .path-card:hover {
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transform: translateY(-1px);
        }

        .path-card h4 {
            margin: 0 0 8px 0;
            font-size: 0.9em;
        }

        .path-card p {
            margin: 0 0 8px 0;
            font-size: 0.8em;
            color: #6c757d;
        }

        .path-meta {
            display: flex;
            gap: 10px;
            font-size: 0.75em;
        }

        /* Quick Actions */
        .quick-actions {
            padding: 15px;
            border-top: 1px solid #dee2e6;
        }

        .quick-actions h3 {
            margin: 0 0 10px 0;
            font-size: 1em;
            color: #495057;
        }

        .quick-actions button {
            width: 100%;
            padding: 8px;
            margin-bottom: 5px;
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            text-align: left;
            cursor: pointer;
            transition: background 0.2s ease;
        }

        .quick-actions button:hover {
            background: #e9ecef;
        }

        /* Breadcrumbs */
        .breadcrumbs {
            padding: 15px 20px;
            background: white;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }

        .breadcrumb-item {
            color: #007bff;
            text-decoration: none;
            font-size: 0.9em;
        }

        .breadcrumb-item:hover {
            text-decoration: underline;
        }

        .breadcrumb-item.active {
            color: #495057;
            font-weight: 500;
        }

        .breadcrumb-separator {
            color: #6c757d;
        }

        /* Progress Tracker */
        .progress-tracker {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px;
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .progress-percentage {
            font-size: 1.2em;
            font-weight: bold;
            color: #007bff;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 15px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            transition: width 0.5s ease;
        }

        .progress-stats {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
        }

        .stat {
            text-align: center;
        }

        .stat-value {
            display: block;
            font-size: 1.5em;
            font-weight: bold;
            color: #495057;
        }

        .stat-label {
            font-size: 0.9em;
            color: #6c757d;
        }

        /* Modal Styles */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 2000;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            background: white;
            border-radius: 8px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }

        .modal-header {
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-close {
            background: none;
            border: none;
            font-size: 1.5em;
            cursor: pointer;
            padding: 5px;
        }

        /* Search Interface */
        .search-interface {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 8px;
            width: 90%;
            max-width: 700px;
            max-height: 80vh;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 2000;
            overflow-y: auto;
        }

        .search-header {
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .search-form {
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }

        .search-form input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 16px;
            margin-bottom: 15px;
        }

        .search-filters {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .filter-group label {
            font-size: 0.9em;
            font-weight: 500;
        }

        .filter-group select,
        .filter-group input[type="checkbox"] {
            padding: 6px;
            border: 1px solid #ced4da;
            border-radius: 4px;
        }

        .search-button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }

        .search-button:hover {
            background: #0056b3;
        }

        .search-results {
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }

        .search-placeholder {
            text-align: center;
            color: #6c757d;
            padding: 40px 20px;
        }

        /* Quick Jump Modal */
        .quick-jump-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 8px;
            width: 90%;
            max-width: 800px;
            max-height: 80vh;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 2000;
            overflow-y: auto;
        }

        .quick-jump-tabs {
            display: flex;
            border-bottom: 1px solid #dee2e6;
        }

        .tab-button {
            flex: 1;
            padding: 15px;
            background: none;
            border: none;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
        }

        .tab-button.active {
            border-bottom-color: #007bff;
            color: #007bff;
            font-weight: 500;
        }

        .quick-jump-content {
            padding: 20px;
        }

        .jump-tab-content {
            display: none;
        }

        .jump-tab-content.active {
            display: block;
        }

        .jump-category {
            margin-bottom: 20px;
        }

        .jump-category h4 {
            margin: 0 0 10px 0;
            color: #495057;
        }

        .jump-sections {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }

        .jump-section {
            padding: 10px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            text-align: left;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .jump-section:hover {
            background: #e9ecef;
            transform: translateY(-1px);
        }

        .jump-section.popular {
            background: #fff3cd;
            border-color: #ffc107;
        }

        /* Keyboard Shortcuts */
        .shortcuts-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 8px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 2000;
            overflow-y: auto;
        }

        .shortcut-category {
            margin-bottom: 20px;
        }

        .shortcut-category h4 {
            margin: 0 0 10px 0;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 5px;
        }

        .shortcuts-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .shortcut-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
        }

        .shortcut-key {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 4px 8px;
            font-family: monospace;
            font-size: 0.9em;
        }

        .shortcut-description {
            color: #6c757d;
            flex: 1;
            margin-left: 15px;
        }

        /* Panels */
        .favorites-panel,
        .recent-panel {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 8px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 2000;
            overflow-y: auto;
        }

        .panel-header {
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .panel-close {
            background: none;
            border: none;
            font-size: 1.5em;
            cursor: pointer;
            padding: 5px;
        }

        .panel-content {
            padding: 20px;
        }

        .favorite-item,
        .recent-item {
            padding: 15px;
            margin-bottom: 10px;
            background: #f8f9fa;
            border-radius: 6px;
        }

        .favorite-item h4,
        .recent-item h4 {
            margin: 0 0 8px 0;
        }

        .favorite-item h4 a,
        .recent-item h4 a {
            color: #007bff;
            text-decoration: none;
        }

        .favorite-item h4 a:hover,
        .recent-item h4 a:hover {
            text-decoration: underline;
        }

        .favorite-item p,
        .recent-item p {
            margin: 0 0 8px 0;
            color: #6c757d;
            font-size: 0.9em;
        }

        .favorite-meta,
        .recent-meta {
            display: flex;
            gap: 10px;
            align-items: center;
            font-size: 0.8em;
        }

        .remove-favorite {
            background: #dc3545;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.8em;
        }

        .no-favorites,
        .no-recent {
            text-align: center;
            color: #6c757d;
            padding: 40px 20px;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .ai-docs-sidebar {
                width: 280px;
            }

            .ai-docs-sidebar.closed {
                transform: translateX(-280px);
            }

            .quick-jump-modal,
            .shortcuts-modal,
            .search-interface {
                width: 95%;
                margin: 20px;
            }

            .jump-sections {
                grid-template-columns: 1fr;
            }
        }

        /* Dark Mode Support */
        @media (prefers-color-scheme: dark) {
            .ai-docs-sidebar {
                background: #2d3748;
                border-right-color: #4a5568;
                color: #e2e8f0;
            }

            .sidebar-header h2 {
                color: #e2e8f0;
            }

            .category-toggle {
                color: #e2e8f0;
            }

            .category-toggle:hover {
                background: #4a5568;
            }

            .nav-section:hover {
                background: #4a5568;
            }

            .nav-section.active {
                background: #2b6cb0;
            }

            .path-card {
                background: #2d3748;
                border-color: #4a5568;
            }

            .quick-actions button {
                background: #2d3748;
                border-color: #4a5568;
                color: #e2e8f0;
            }

            .quick-actions button:hover {
                background: #4a5568;
            }
        }
        """


def main():
    """Demo the navigation UI components."""
    from navigation_system import AIDocumentationNavigator

    navigator = AIDocumentationNavigator()
    ui_renderer = NavigationUIRenderer(navigator)

    print("=== AI Documentation Navigation UI Demo ===\n")

    # Example sidebar rendering
    print("1. Sidebar Navigation Component:")
    sidebar_html = ui_renderer.render_sidebar("01_foundational_ml")
    print(f"Generated {len(sidebar_html)} characters of HTML")

    # Example breadcrumbs
    print("\n2. Breadcrumbs Component:")
    breadcrumbs_html = ui_renderer.render_breadcrumbs("01_foundational_ml")
    print(f"Generated {len(breadcrumbs_html)} characters of HTML")

    # Example search interface
    print("\n3. Search Interface Component:")
    search_html = ui_renderer.render_search_interface()
    print(f"Generated {len(search_html)} characters of HTML")

    # Example JavaScript
    print("\n4. JavaScript Interactions:")
    js_code = ui_renderer.generate_javascript()
    print(f"Generated {len(js_code)} characters of JavaScript")

    # Example CSS
    print("\n5. CSS Styles:")
    css_code = ui_renderer.generate_css()
    print(f"Generated {len(css_code)} characters of CSS")

    print("\nNavigation UI components generated successfully!")


if __name__ == "__main__":
    main()