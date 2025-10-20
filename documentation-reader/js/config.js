// Configuration and Constants
window.AIDocsReader = window.AIDocsReader || {};

AIDocsReader.Config = {
    // Documentation Structure
    sections: [
        {
            id: '01_Foundational_Machine_Learning',
            title: 'Foundational Machine Learning',
            icon: 'fas fa-brain',
            description: 'Mathematical foundations and core ML concepts',
            difficulty: 'beginner',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics',
                '05_Exercises_Projects'
            ]
        },
        {
            id: '02_Advanced_Deep_Learning',
            title: 'Advanced Deep Learning',
            icon: 'fas fa-network-wired',
            description: 'Neural architectures and specialized systems',
            difficulty: 'intermediate',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '03_Natural_Language_Processing',
            title: 'Natural Language Processing',
            icon: 'fas fa-language',
            description: 'Text processing, language models, and LLM applications',
            difficulty: 'intermediate',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '04_Computer_Vision',
            title: 'Computer Vision',
            icon: 'fas fa-eye',
            description: 'Image processing, object detection, and visual AI',
            difficulty: 'intermediate',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '05_Generative_AI',
            title: 'Generative AI',
            icon: 'fas fa-palette',
            description: 'Foundation models and creative applications',
            difficulty: 'advanced',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '06_AI_Agents_and_Autonomous',
            title: 'AI Agents and Autonomous Systems',
            icon: 'fas fa-robot',
            description: 'Autonomous agents and reinforcement learning',
            difficulty: 'advanced',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '07_AI_Ethics_and_Safety',
            title: 'AI Ethics and Safety',
            icon: 'fas fa-shield-alt',
            description: 'Ethical considerations and safety research',
            difficulty: 'intermediate',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '08_AI_Applications_Industry',
            title: 'AI Applications in Industry',
            icon: 'fas fa-industry',
            description: 'Real-world applications and industry case studies',
            difficulty: 'intermediate',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '09_Emerging_Interdisciplinary',
            title: 'Emerging Interdisciplinary AI',
            icon: 'fas fa-atom',
            description: 'AI combined with other scientific fields',
            difficulty: 'advanced',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '10_Technical_Methodological',
            title: 'Technical Methodological',
            icon: 'fas fa-cogs',
            description: 'AI systems, hardware, and development tools',
            difficulty: 'advanced',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '11_Future_Directions',
            title: 'Future Directions',
            icon: 'fas fa-rocket',
            description: 'AGI, consciousness, and long-term research',
            difficulty: 'expert',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        },
        {
            id: '12_Emerging_Research_2025',
            title: 'Emerging Research 2025',
            icon: 'fas fa-flask',
            description: 'Latest research and emerging trends',
            difficulty: 'expert',
            modules: [
                '01_Theory_Foundations',
                '02_Practical_Implementations',
                '03_Case_Studies',
                '04_Advanced_Topics'
            ]
        }
    ],

    // Learning Paths
    learningPaths: {
        beginner: {
            name: 'Beginner Path',
            description: 'Start with the fundamentals of AI and machine learning',
            sections: [
                '01_Foundational_Machine_Learning',
                '02_Advanced_Deep_Learning'
            ],
            estimatedTime: '40 hours',
            color: '#10b981'
        },
        intermediate: {
            name: 'Intermediate Path',
            description: 'Build on fundamentals with practical applications',
            sections: [
                '02_Advanced_Deep_Learning',
                '03_Natural_Language_Processing',
                '04_Computer_Vision',
                '05_Generative_AI'
            ],
            estimatedTime: '60 hours',
            color: '#3b82f6'
        },
        advanced: {
            name: 'Advanced Path',
            description: 'Master complex AI concepts and applications',
            sections: [
                '05_Generative_AI',
                '06_AI_Agents_and_Autonomous',
                '09_Emerging_Interdisciplinary',
                '10_Technical_Methodological'
            ],
            estimatedTime: '80 hours',
            color: '#8b5cf6'
        },
        researcher: {
            name: 'Researcher Path',
            description: 'Focus on cutting-edge research and future directions',
            sections: [
                '07_AI_Ethics_and_Safety',
                '11_Future_Directions',
                '12_Emerging_Research_2025'
            ],
            estimatedTime: '50 hours',
            color: '#ef4444'
        },
        practitioner: {
            name: 'Industry Practitioner',
            description: 'Real-world applications and industry focus',
            sections: [
                '08_AI_Applications_Industry',
                '10_Technical_Methodological',
                '12_Emerging_Research_2025'
            ],
            estimatedTime: '45 hours',
            color: '#f59e0b'
        }
    },

    // Content Types
    contentTypes: {
        markdown: {
            extensions: ['.md'],
            icon: 'fas fa-file-alt',
            color: '#2563eb'
        },
        notebook: {
            extensions: ['.ipynb'],
            icon: 'fas fa-jupyter',
            color: '#f59e0b'
        },
        pdf: {
            extensions: ['.pdf'],
            icon: 'fas fa-file-pdf',
            color: '#ef4444'
        },
        code: {
            extensions: ['.py', '.js', '.ts', '.java', '.cpp'],
            icon: 'fas fa-code',
            color: '#10b981'
        }
    },

    // Difficulty Levels
    difficultyLevels: {
        beginner: {
            color: '#10b981',
            icon: 'fas fa-seedling',
            description: 'New to AI concepts'
        },
        intermediate: {
            color: '#3b82f6',
            icon: 'fas fa-chart-line',
            description: 'Some AI knowledge'
        },
        advanced: {
            color: '#8b5cf6',
            icon: 'fas fa-mountain',
            description: 'Strong AI background'
        },
        expert: {
            color: '#ef4444',
            icon: 'fas fa-crown',
            description: 'AI professional/researcher'
        }
    },

    // Reading Speed (words per minute)
    readingSpeeds: {
        slow: 150,
        normal: 200,
        fast: 250
    },

    // Storage Keys
    storageKeys: {
        theme: 'ai-docs-theme',
        bookmarks: 'ai-docs-bookmarks',
        progress: 'ai-docs-progress',
        notes: 'ai-docs-notes',
        settings: 'ai-docs-settings',
        readingHistory: 'ai-docs-history',
        lastPosition: 'ai-docs-last-position'
    },

    // API Endpoints
    api: {
        content: '../',
        search: '../api/search',
        analytics: '../api/analytics'
    },

    // Default Settings
    defaultSettings: {
        theme: 'light',
        fontSize: 16,
        fontFamily: 'Inter',
        lineHeight: 1.6,
        maxWidth: 900,
        autoSaveProgress: true,
        showRelatedContent: true,
        enableNotifications: false,
        readingSpeed: 'normal',
        distractionFreeMode: false
    },

    // Achievement System
    achievements: {
        first_read: {
            name: 'First Steps',
            description: 'Read your first documentation topic',
            icon: 'fas fa-book-open',
            color: '#10b981'
        },
        marathon_reader: {
            name: 'Marathon Reader',
            description: 'Read for over 3 hours total',
            icon: 'fas fa-clock',
            color: '#f59e0b'
        },
        completionist: {
            name: 'Completionist',
            description: 'Complete an entire section',
            icon: 'fas fa-trophy',
            color: '#8b5cf6'
        },
        note_taker: {
            name: 'Note Taker',
            description: 'Create 10 notes',
            icon: 'fas fa-sticky-note',
            color: '#3b82f6'
        },
        explorer: {
            name: 'Explorer',
            description: 'Visit all main sections',
            icon: 'fas fa-compass',
            color: '#ef4444'
        },
        night_owl: {
            name: 'Night Owl',
            description: 'Read for 1 hour after 10 PM',
            icon: 'fas fa-moon',
            color: '#6d28d9'
        }
    },

    // Keyboard Shortcuts
    shortcuts: {
        'ctrl+k': 'openSearch',
        'ctrl+/': 'toggleSidebar',
        'ctrl+b': 'toggleBookmark',
        'ctrl+n': 'addNote',
        'ctrl+shift+d': 'toggleDistractionFree',
        'ctrl+shift+t': 'toggleTheme',
        'escape': 'closeModal',
        'arrow-left': 'navigatePrevious',
        'arrow-right': 'navigateNext',
        'ctrl+plus': 'increaseFontSize',
        'ctrl+minus': 'decreaseFontSize'
    }
};

// Utility Functions
AIDocsReader.Utils = {
    // Format time duration
    formatDuration: function(minutes) {
        if (minutes < 60) {
            return `${Math.round(minutes)} min`;
        }
        const hours = Math.floor(minutes / 60);
        const mins = Math.round(minutes % 60);
        return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
    },

    // Format reading time based on word count
    formatReadingTime: function(wordCount, speed = 'normal') {
        const wpm = AIDocsReader.Config.readingSpeeds[speed];
        const minutes = wordCount / wpm;
        return this.formatDuration(minutes);
    },

    // Get difficulty color
    getDifficultyColor: function(level) {
        return AIDocsReader.Config.difficultyLevels[level]?.color || '#6b7280';
    },

    // Get section by ID
    getSection: function(sectionId) {
        return AIDocsReader.Config.sections.find(s => s.id === sectionId);
    },

    // Get learning path by ID
    getLearningPath: function(pathId) {
        return AIDocsReader.Config.learningPaths[pathId];
    },

    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Throttle function
    throttle: function(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    // Sanitize HTML
    sanitizeHtml: function(html) {
        const div = document.createElement('div');
        div.textContent = html;
        return div.innerHTML;
    },

    // Generate unique ID
    generateId: function() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    },

    // Calculate reading progress
    calculateProgress: function(read, total) {
        if (total === 0) return 0;
        return Math.round((read / total) * 100);
    },

    // Get estimated reading time for content
    estimateReadingTime: function(content) {
        const wordsPerMinute = 200;
        const words = content.trim().split(/\s+/).length;
        return Math.ceil(words / wordsPerMinute);
    }
};