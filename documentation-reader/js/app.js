// Main Application Controller
AIDocsReader.App = {
    // Application state
    state: {
        currentSection: null,
        currentModule: null,
        currentContent: null,
        isLoading: false,
        sidebarOpen: true,
        theme: 'light',
        bookmarks: [],
        progress: {},
        notes: [],
        settings: {}
    },

    // Initialize application
    init: function() {
        console.log('Initializing AI Documentation Reader...');

        // Initialize modules
        this.initStorage();
        this.initRenderer();
        this.initTheme();
        this.initEventListeners();
        this.initKeyboardShortcuts();
        this.loadInitialContent();

        // Hide loading screen
        this.hideLoadingScreen();

        console.log('Application initialized successfully');
    },

    // Initialize storage
    initStorage: function() {
        AIDocsReader.Storage.init();
        this.state.settings = AIDocsReader.Storage.getSettings();
        this.state.bookmarks = AIDocsReader.Storage.getBookmarks();
        this.state.progress = AIDocsReader.Storage.getProgress();
        this.state.notes = AIDocsReader.Storage.getNotes();
        this.state.theme = this.state.settings.theme || 'light';
    },

    // Initialize renderer
    initRenderer: function() {
        AIDocsReader.Renderer.init();
    },

    // Initialize theme
    initTheme: function() {
        AIDocsReader.Themes.init();
        AIDocsReader.Themes.setTheme(this.state.theme);
    },

    // Initialize event listeners
    initEventListeners: function() {
        // Sidebar controls
        document.getElementById('sidebarToggle').addEventListener('click', () => this.toggleSidebar());
        document.getElementById('sidebarClose').addEventListener('click', () => this.closeSidebar());

        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', () => this.toggleTheme());

        // Search
        const searchInput = document.getElementById('globalSearch');
        searchInput.addEventListener('input', AIDocsReader.Utils.debounce((e) => {
            AIDocsReader.Search.performSearch(e.target.value);
        }, 300));

        // Learning path selector
        document.getElementById('pathSelect').addEventListener('change', (e) => {
            this.selectLearningPath(e.target.value);
        });

        // Content controls
        document.getElementById('fontIncrease').addEventListener('click', () => this.adjustFontSize(1));
        document.getElementById('fontDecrease').addEventListener('click', () => this.adjustFontSize(-1));
        document.getElementById('distractionFree').addEventListener('click', () => this.toggleDistractionFree());

        // Bookmark controls
        document.getElementById('bookmarkBtn').addEventListener('click', () => this.openBookmarksModal());
        document.getElementById('toggleBookmark').addEventListener('click', () => this.toggleCurrentBookmark());

        // Progress controls
        document.getElementById('progressBtn').addEventListener('click', () => this.openProgressModal());

        // Notes controls
        document.getElementById('notesBtn').addEventListener('click', () => this.openNotesModal());
        document.getElementById('addNoteBtn').addEventListener('click', () => this.openNotesEditor());

        // Quick actions
        document.getElementById('continueReading').addEventListener('click', () => this.continueReading());
        document.getElementById('randomTopic').addEventListener('click', () => this.loadRandomTopic());
        document.getElementById('bookmarksList').addEventListener('click', () => this.openBookmarksModal());

        // Modal controls
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modalId = e.target.closest('.modal-close').dataset.modal;
                this.closeModal(modalId);
            });
        });

        // Click outside modal to close
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeModal(modal.id);
                }
            });
        });

        // Initialize navigation
        this.initializeNavigation();

        // Initialize progress tracking
        AIDocsReader.Progress.init();
    },

    // Initialize keyboard shortcuts
    initKeyboardShortcuts: function() {
        document.addEventListener('keydown', (e) => {
            const key = this.getShortcutKey(e);

            if (AIDocsReader.Config.shortcuts[key]) {
                e.preventDefault();
                this.handleShortcut(AIDocsReader.Config.shortcuts[key]);
            }
        });
    },

    // Get shortcut key string
    getShortcutKey: function(e) {
        const parts = [];
        if (e.ctrlKey || e.metaKey) parts.push('ctrl');
        if (e.shiftKey) parts.push('shift');
        if (e.altKey) parts.push('alt');
        if (e.key && !['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) {
            parts.push(e.key.toLowerCase());
        }
        return parts.join('+');
    },

    // Handle keyboard shortcut
    handleShortcut: function(action) {
        switch (action) {
            case 'openSearch':
                document.getElementById('globalSearch').focus();
                break;
            case 'toggleSidebar':
                this.toggleSidebar();
                break;
            case 'toggleBookmark':
                this.toggleCurrentBookmark();
                break;
            case 'addNote':
                this.openNotesEditor();
                break;
            case 'toggleDistractionFree':
                this.toggleDistractionFree();
                break;
            case 'toggleTheme':
                this.toggleTheme();
                break;
            case 'closeModal':
                this.closeAllModals();
                break;
            case 'increaseFontSize':
                this.adjustFontSize(1);
                break;
            case 'decreaseFontSize':
                this.adjustFontSize(-1);
                break;
            case 'navigatePrevious':
                this.navigatePrevious();
                break;
            case 'navigateNext':
                this.navigateNext();
                break;
        }
    },

    // Initialize navigation
    initializeNavigation: function() {
        const sectionNav = document.getElementById('sectionNav');
        const sections = AIDocsReader.Config.sections;

        const navHtml = sections.map(section => `
            <a href="#" class="nav-item" data-section="${section.id}">
                <i class="${section.icon}"></i>
                <span>${section.title}</span>
            </a>
        `).join('');

        sectionNav.innerHTML = navHtml;

        // Add click handlers
        sectionNav.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const sectionId = item.dataset.section;
                this.loadSection(sectionId);
            });
        });
    },

    // Load initial content
    loadInitialContent: function() {
        const lastPosition = AIDocsReader.Storage.getLastPosition();
        if (lastPosition) {
            this.loadContent(lastPosition.sectionId, lastPosition.moduleId, lastPosition.contentId);
        } else {
            this.showWelcomeScreen();
        }
    },

    // Show welcome screen
    showWelcomeScreen: function() {
        const contentDisplay = document.getElementById('markdownContent');
        contentDisplay.innerHTML = `
            <div class="welcome-screen">
                <div class="welcome-header">
                    <i class="fas fa-graduation-cap" style="font-size: 4rem; color: var(--primary-color);"></i>
                    <h1>Welcome to AI Documentation Reader</h1>
                    <p class="welcome-subtitle">Your comprehensive learning platform for Artificial Intelligence</p>
                </div>

                <div class="welcome-stats">
                    <div class="stat-card">
                        <i class="fas fa-book"></i>
                        <div class="stat-value">25+</div>
                        <div class="stat-label">Main Sections</div>
                    </div>
                    <div class="stat-card">
                        <i class="fas fa-file-alt"></i>
                        <div class="stat-value">1500+</div>
                        <div class="stat-label">Topics</div>
                    </div>
                    <div class="stat-card">
                        <i class="fas fa-code"></i>
                        <div class="stat-value">75+</div>
                        <div class="stat-label">Interactive Notebooks</div>
                    </div>
                    <div class="stat-card">
                        <i class="fas fa-clock"></i>
                        <div class="stat-value">200+</div>
                        <div class="stat-label">Hours of Content</div>
                    </div>
                </div>

                <div class="welcome-features">
                    <h2>Features</h2>
                    <div class="features-grid">
                        <div class="feature-card">
                            <i class="fas fa-route"></i>
                            <h3>Guided Learning Paths</h3>
                            <p>Structured curriculum from beginner to expert level</p>
                        </div>
                        <div class="feature-card">
                            <i class="fas fa-chart-line"></i>
                            <h3>Progress Tracking</h3>
                            <p>Monitor your learning journey with detailed analytics</p>
                        </div>
                        <div class="feature-card">
                            <i class="fas fa-bookmark"></i>
                            <h3>Smart Bookmarks</h3>
                            <p>Save and organize important topics for quick access</p>
                        </div>
                        <div class="feature-card">
                            <i class="fas fa-sticky-note"></i>
                            <h3>Personal Notes</h3>
                            <p>Take notes directly in the interface for reference</p>
                        </div>
                        <div class="feature-card">
                            <i class="fas fa-moon"></i>
                            <h3>Dark Mode</h3>
                            <p>Comfortable reading experience with theme options</p>
                        </div>
                        <div class="feature-card">
                            <i class="fas fa-search"></i>
                            <h3>Smart Search</h3>
                            <p>Quickly find topics across all documentation</p>
                        </div>
                    </div>
                </div>

                <div class="welcome-actions">
                    <button class="btn btn-primary btn-lg" onclick="AIDocsReader.App.showLearningPaths()">
                        <i class="fas fa-play"></i>
                        Start Learning
                    </button>
                    <button class="btn btn-secondary btn-lg" onclick="AIDocsReader.App.browseSections()">
                        <i class="fas fa-th-list"></i>
                        Browse Sections
                    </button>
                </div>

                <div class="welcome-tips">
                    <h3>Quick Tips</h3>
                    <ul>
                        <li>Press <kbd>Ctrl+K</kbd> to quickly search for topics</li>
                        <li>Use <kbd>Ctrl+D</kbd> to toggle distraction-free mode</li>
                        <li>Press <kbd>Ctrl+B</kbd> to bookmark current content</li>
                        <li>Your progress is automatically saved as you read</li>
                    </ul>
                </div>
            </div>
        `;
    },

    // Show learning paths
    showLearningPaths: function() {
        const contentDisplay = document.getElementById('markdownContent');
        const paths = AIDocsReader.Config.learningPaths;

        const pathsHtml = Object.entries(paths).map(([key, path]) => `
            <div class="learning-path-card" style="border-left: 4px solid ${path.color};">
                <div class="path-header">
                    <h3>${path.name}</h3>
                    <span class="path-duration" style="color: ${path.color};">
                        <i class="fas fa-clock"></i> ${path.estimatedTime}
                    </span>
                </div>
                <p class="path-description">${path.description}</p>
                <div class="path-sections">
                    ${path.sections.map(sectionId => {
                        const section = AIDocsReader.Utils.getSection(sectionId);
                        return section ? `<span class="path-section">${section.title}</span>` : '';
                    }).join('')}
                </div>
                <button class="btn btn-primary" onclick="AIDocsReader.App.selectLearningPath('${key}')">
                    Start This Path
                </button>
            </div>
        `).join('');

        contentDisplay.innerHTML = `
            <div class="learning-paths">
                <div class="page-header">
                    <h1>Learning Paths</h1>
                    <p>Choose a guided learning path that matches your skill level and goals</p>
                </div>
                <div class="paths-grid">
                    ${pathsHtml}
                </div>
            </div>
        `;
    },

    // Browse sections
    browseSections: function() {
        const sections = AIDocsReader.Config.sections;
        this.renderSectionGrid(sections);
    },

    // Render section grid
    renderSectionGrid: function(sections) {
        const contentDisplay = document.getElementById('markdownContent');

        const sectionsHtml = sections.map(section => `
            <div class="section-card" onclick="AIDocsReader.App.loadSection('${section.id}')">
                <div class="section-icon">
                    <i class="${section.icon}"></i>
                </div>
                <div class="section-content">
                    <h3>${section.title}</h3>
                    <p>${section.description}</p>
                    <div class="section-meta">
                        <span class="difficulty-badge ${section.difficulty}">${section.difficulty}</span>
                        <span class="module-count">${section.modules.length} modules</span>
                    </div>
                </div>
            </div>
        `).join('');

        contentDisplay.innerHTML = `
            <div class="sections-browser">
                <div class="page-header">
                    <h1>Browse Sections</h1>
                    <p>Explore our comprehensive AI documentation sections</p>
                </div>
                <div class="sections-grid">
                    ${sectionsHtml}
                </div>
            </div>
        `;
    },

    // Select learning path
    selectLearningPath: function(pathId) {
        const path = AIDocsReader.Utils.getLearningPath(pathId);
        if (!path) return;

        // Update path selector
        document.getElementById('pathSelect').value = pathId;

        // Filter sections
        const filteredSections = path.sections.map(id =>
            AIDocsReader.Utils.getSection(id)
        ).filter(Boolean);

        this.renderSectionGrid(filteredSections);
        this.updateBreadcrumb('Learning Paths', path.name);
    },

    // Load section
    loadSection: function(sectionId) {
        const section = AIDocsReader.Utils.getSection(sectionId);
        if (!section) return;

        this.state.currentSection = section;
        this.updateBreadcrumb('Sections', section.title);

        // Update active navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionId}"]`)?.classList.add('active');

        // Load section overview
        this.loadSectionOverview(section);
    },

    // Load section overview
    loadSectionOverview: function(section) {
        const contentDisplay = document.getElementById('markdownContent');

        // Get progress for this section
        const sectionProgress = this.getSectionProgress(section.id);

        const overviewHtml = `
            <div class="section-overview">
                <div class="section-header">
                    <div class="section-icon-large">
                        <i class="${section.icon}"></i>
                    </div>
                    <div class="section-info">
                        <h1>${section.title}</h1>
                        <p class="section-description">${section.description}</p>
                        <div class="section-meta">
                            <span class="difficulty-badge ${section.difficulty}">${section.difficulty}</span>
                            <span class="module-count">${section.modules.length} modules</span>
                            <span class="progress-indicator">
                                <i class="fas fa-chart-line"></i>
                                ${sectionProgress}% Complete
                            </span>
                        </div>
                    </div>
                </div>

                <div class="section-progress-bar">
                    <div class="progress-fill" style="width: ${sectionProgress}%"></div>
                </div>

                <div class="modules-grid">
                    ${section.modules.map((module, index) => `
                        <div class="module-card" onclick="AIDocsReader.App.loadModule('${section.id}', '${module}')">
                            <div class="module-number">${index + 1}</div>
                            <div class="module-content">
                                <h3>${this.formatModuleTitle(module)}</h3>
                                <p>${this.getModuleDescription(module)}</p>
                                <div class="module-status">
                                    ${this.getModuleProgress(section.id, module)}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>

                <div class="section-actions">
                    <button class="btn btn-primary" onclick="AIDocsReader.App.startSection('${section.id}')">
                        <i class="fas fa-play"></i>
                        Start Section
                    </button>
                    <button class="btn btn-secondary" onclick="AIDocsReader.App.continueSection('${section.id}')">
                        <i class="fas fa-redo"></i>
                        Continue Learning
                    </button>
                </div>
            </div>
        `;

        contentDisplay.innerHTML = overviewHtml;

        // Save position
        this.savePosition(section.id, null, null);
    },

    // Format module title
    formatModuleTitle: function(moduleId) {
        return moduleId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    },

    // Get module description
    getModuleDescription: function(moduleId) {
        const descriptions = {
            '01_Theory_Foundations': 'Mathematical foundations and theoretical concepts',
            '02_Practical_Implementations': 'Hands-on code examples and implementations',
            '03_Case_Studies': 'Real-world applications and examples',
            '04_Advanced_Topics': 'Cutting-edge research and advanced concepts',
            '05_Exercises_Projects': 'Practice exercises and projects'
        };
        return descriptions[moduleId] || 'Learn about this topic in detail';
    },

    // Get module progress
    getModuleProgress: function(sectionId, moduleId) {
        const progress = AIDocsReader.Storage.getContentProgress(`${sectionId}/${moduleId}`);
        if (progress && progress.percentage >= 100) {
            return '<span class="status-complete"><i class="fas fa-check"></i> Completed</span>';
        } else if (progress && progress.percentage > 0) {
            return `<span class="status-progress">${progress.percentage}% Complete</span>`;
        }
        return '<span class="status-new"><i class="fas fa-circle"></i> New</span>';
    },

    // Get section progress
    getSectionProgress: function(sectionId) {
        const section = AIDocsReader.Utils.getSection(sectionId);
        if (!section) return 0;

        let totalModules = section.modules.length;
        let completedModules = 0;

        section.modules.forEach(module => {
            const progress = AIDocsReader.Storage.getContentProgress(`${sectionId}/${module}`);
            if (progress && progress.percentage >= 100) {
                completedModules++;
            }
        });

        return Math.round((completedModules / totalModules) * 100);
    },

    // Load module
    loadModule: function(sectionId, moduleId) {
        // For now, show a placeholder
        const contentDisplay = document.getElementById('markdownContent');
        contentDisplay.innerHTML = `
            <div class="module-loading">
                <div class="loading-spinner"></div>
                <h2>Loading Module Content</h2>
                <p>Preparing ${this.formatModuleTitle(moduleId)} from ${AIDocsReader.Utils.getSection(sectionId)?.title}</p>
                <p class="loading-note">This will load the actual markdown content from the documentation files.</p>
            </div>
        `;

        this.updateBreadcrumb(AIDocsReader.Utils.getSection(sectionId)?.title, this.formatModuleTitle(moduleId));
        this.savePosition(sectionId, moduleId, null);
    },

    // Start section
    startSection: function(sectionId) {
        const section = AIDocsReader.Utils.getSection(sectionId);
        if (section && section.modules.length > 0) {
            this.loadModule(sectionId, section.modules[0]);
        }
    },

    // Continue section
    continueSection: function(sectionId) {
        const section = AIDocsReader.Utils.getSection(sectionId);
        if (!section) return;

        // Find first incomplete module
        for (const module of section.modules) {
            const progress = AIDocsReader.Storage.getContentProgress(`${sectionId}/${module}`);
            if (!progress || progress.percentage < 100) {
                this.loadModule(sectionId, module);
                return;
            }
        }

        // All modules completed
        this.loadModule(sectionId, section.modules[0]);
    },

    // Continue reading (from last position)
    continueReading: function() {
        const lastPosition = AIDocsReader.Storage.getLastPosition();
        if (lastPosition) {
            if (lastPosition.moduleId) {
                this.loadModule(lastPosition.sectionId, lastPosition.moduleId);
            } else {
                this.loadSection(lastPosition.sectionId);
            }
        } else {
            this.showWelcomeScreen();
        }
    },

    // Load random topic
    loadRandomTopic: function() {
        const sections = AIDocsReader.Config.sections;
        const randomSection = sections[Math.floor(Math.random() * sections.length)];
        this.loadSection(randomSection.id);
    },

    // Toggle sidebar
    toggleSidebar: function() {
        const sidebar = document.getElementById('sidebar');
        this.state.sidebarOpen = !this.state.sidebarOpen;

        if (this.state.sidebarOpen) {
            sidebar.classList.add('open');
        } else {
            sidebar.classList.remove('open');
        }
    },

    // Close sidebar
    closeSidebar: function() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.remove('open');
        this.state.sidebarOpen = false;
    },

    // Toggle theme
    toggleTheme: function() {
        const currentTheme = this.state.theme;
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        AIDocsReader.Themes.setTheme(newTheme);
        this.state.theme = newTheme;
        AIDocsReader.Storage.updateSetting('theme', newTheme);
    },

    // Toggle distraction free mode
    toggleDistractionFree: function() {
        document.body.classList.toggle('distraction-free');
        this.state.settings.distractionFreeMode = !this.state.settings.distractionFreeMode;
        AIDocsReader.Storage.updateSetting('distractionFreeMode', this.state.settings.distractionFreeMode);
    },

    // Adjust font size
    adjustFontSize: function(delta) {
        const currentSize = parseInt(this.state.settings.fontSize) || 16;
        const newSize = Math.max(12, Math.min(24, currentSize + delta));

        document.documentElement.style.setProperty('--base-font-size', newSize + 'px');
        this.state.settings.fontSize = newSize;
        AIDocsReader.Storage.updateSetting('fontSize', newSize);
    },

    // Update breadcrumb
    updateBreadcrumb: function(...items) {
        const breadcrumb = document.getElementById('breadcrumb');
        const itemsHtml = [
            '<a href="#" class="breadcrumb-item" data-section="home"><i class="fas fa-home"></i> Home</a>'
        ];

        items.forEach((item, index) => {
            if (index === items.length - 1) {
                itemsHtml.push(`<span class="breadcrumb-item current">${item}</span>`);
            } else {
                itemsHtml.push(`<a href="#" class="breadcrumb-item">${item}</a>`);
            }
        });

        breadcrumb.innerHTML = itemsHtml.join('');
    },

    // Save position
    savePosition: function(sectionId, moduleId, contentId) {
        AIDocsReader.Storage.setLastPosition({
            sectionId,
            moduleId,
            contentId,
            timestamp: Date.now()
        });
    },

    // Modal management
    openModal: function(modalId) {
        document.getElementById(modalId).classList.add('active');
    },

    closeModal: function(modalId) {
        document.getElementById(modalId).classList.remove('active');
    },

    closeAllModals: function() {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
        });
    },

    // Bookmark methods
    toggleCurrentBookmark: function() {
        // Implementation will depend on current content
        console.log('Toggle bookmark for current content');
    },

    openBookmarksModal: function() {
        AIDocsReader.Bookmarks.showBookmarksModal();
    },

    // Progress methods
    openProgressModal: function() {
        AIDocsReader.Progress.showProgressModal();
    },

    // Notes methods
    openNotesModal: function() {
        AIDocsReader.Notes.showNotesModal();
    },

    openNotesEditor: function() {
        AIDocsReader.Notes.openNotesEditor();
    },

    // Navigation methods
    navigatePrevious: function() {
        // Implementation for previous navigation
        console.log('Navigate to previous');
    },

    navigateNext: function() {
        // Implementation for next navigation
        console.log('Navigate to next');
    },

    // Hide loading screen
    hideLoadingScreen: function() {
        setTimeout(() => {
            const loadingScreen = document.getElementById('loadingScreen');
            loadingScreen.classList.add('hidden');

            // Animate progress fill
            const progressFill = loadingScreen.querySelector('.progress-fill');
            if (progressFill) {
                progressFill.style.width = '100%';
            }
        }, 1500);
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    AIDocsReader.App.init();
});