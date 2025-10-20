// Search Controller
AIDocsReader.Search = {
    // Initialize search
    init: function() {
        this.setupSearchListeners();
        this.searchIndex = null;
        this.buildSearchIndex();
    },

    // Setup search listeners
    setupSearchListeners: function() {
        const searchInput = document.getElementById('globalSearch');

        if (searchInput) {
            // Search on input with debouncing
            searchInput.addEventListener('input', AIDocsReader.Utils.debounce((e) => {
                const query = e.target.value.trim();
                this.performSearch(query);
            }, 300));

            // Handle Enter key
            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.handleSearchSubmit(e.target.value.trim());
                }
            });

            // Handle escape key
            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.clearSearch();
                    searchInput.blur();
                }
            });

            // Focus search on Ctrl+K
            document.addEventListener('keydown', (e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                    e.preventDefault();
                    searchInput.focus();
                    searchInput.select();
                }
            });
        }

        // Close search results when clicking outside
        document.addEventListener('click', (e) => {
            const searchContainer = document.querySelector('.search-container');
            const searchResults = document.getElementById('searchResults');

            if (searchContainer && !searchContainer.contains(e.target) && searchResults) {
                this.hideSearchResults();
            }
        });
    },

    // Build search index
    buildSearchIndex: function() {
        // In a real implementation, this would index all available content
        // For now, we'll create a mock index based on sections
        const sections = AIDocsReader.Config.sections;
        this.searchIndex = sections.map(section => ({
            id: section.id,
            title: section.title,
            description: section.description,
            type: 'section',
            section: section.title,
            difficulty: section.difficulty,
            url: `#section/${section.id}`,
            content: `${section.title} ${section.description} ${section.difficulty}`,
            tags: [section.difficulty, ...section.modules]
        }));

        console.log('Search index built with', this.searchIndex.length, 'items');
    },

    // Perform search
    performSearch: function(query) {
        if (!query || query.length < 2) {
            this.hideSearchResults();
            return;
        }

        const results = this.searchInIndex(query);
        this.displaySearchResults(results, query);
    },

    // Search in index
    searchInIndex: function(query) {
        if (!this.searchIndex) return [];

        const normalizedQuery = query.toLowerCase();
        const results = [];

        this.searchIndex.forEach(item => {
            const score = this.calculateRelevanceScore(item, normalizedQuery);
            if (score > 0) {
                results.push({
                    ...item,
                    score,
                    highlights: this.generateHighlights(item, normalizedQuery)
                });
            }
        });

        // Sort by relevance score
        results.sort((a, b) => b.score - a.score);

        return results.slice(0, 10); // Limit to 10 results
    },

    // Calculate relevance score
    calculateRelevanceScore: function(item, query) {
        let score = 0;
        const titleLower = item.title.toLowerCase();
        const descriptionLower = item.description.toLowerCase();
        const contentLower = item.content.toLowerCase();

        // Exact title match gets highest score
        if (titleLower === query) score += 100;
        else if (titleLower.includes(query)) score += 50;

        // Description matches
        if (descriptionLower.includes(query)) score += 25;

        // Content matches
        if (contentLower.includes(query)) score += 10;

        // Word boundary matches
        const words = query.split(' ');
        words.forEach(word => {
            if (word.length > 2) {
                if (titleLower.includes(word)) score += 15;
                if (descriptionLower.includes(word)) score += 8;
                if (contentLower.includes(word)) score += 3;
            }
        });

        // Bonus for matching difficulty level
        if (item.difficulty && item.difficulty.toLowerCase().includes(query)) {
            score += 20;
        }

        return score;
    },

    // Generate highlights
    generateHighlights: function(item, query) {
        const highlights = [];
        const words = query.split(' ').filter(word => word.length > 2);

        // Highlight in title
        const titleHighlight = this.highlightText(item.title, words);
        if (titleHighlight !== item.title) {
            highlights.title = titleHighlight;
        }

        // Highlight in description
        const descHighlight = this.highlightText(item.description, words);
        if (descHighlight !== item.description) {
            highlights.description = descHighlight;
        }

        return highlights;
    },

    // Highlight text
    highlightText: function(text, words) {
        let highlighted = text;

        words.forEach(word => {
            const regex = new RegExp(`(${word})`, 'gi');
            highlighted = highlighted.replace(regex, '<mark>$1</mark>');
        });

        return highlighted;
    },

    // Display search results
    displaySearchResults: function(results, query) {
        const searchResults = document.getElementById('searchResults');
        if (!searchResults) return;

        if (results.length === 0) {
            searchResults.innerHTML = `
                <div class="search-no-results">
                    <i class="fas fa-search"></i>
                    <p>No results found for "${query}"</p>
                    <small>Try different keywords or check spelling</small>
                </div>
            `;
        } else {
            const resultsHtml = results.map(result => this.renderSearchResult(result, query)).join('');
            searchResults.innerHTML = `
                <div class="search-results-header">
                    <span>${results.length} result${results.length !== 1 ? 's' : ''} for "${query}"</span>
                </div>
                <div class="search-results-list">
                    ${resultsHtml}
                </div>
            `;
        }

        // Show results
        this.showSearchResults();
    },

    // Render search result
    renderSearchResult: function(result, query) {
        const highlights = result.highlights || {};
        const typeIcon = this.getTypeIcon(result.type);
        const difficultyBadge = `<span class="difficulty-badge ${result.difficulty}">${result.difficulty}</span>`;

        return `
            <div class="search-result-item" onclick="AIDocsReader.Search.selectSearchResult('${result.id}')">
                <div class="search-result-header">
                    <div class="search-result-icon">
                        <i class="${typeIcon}"></i>
                    </div>
                    <div class="search-result-title">
                        ${highlights.title || result.title}
                    </div>
                    <div class="search-result-meta">
                        ${difficultyBadge}
                        <span class="search-result-type">${result.type}</span>
                    </div>
                </div>
                <div class="search-result-description">
                    ${highlights.description || result.description}
                </div>
                <div class="search-result-footer">
                    <span class="search-result-section">${result.section}</span>
                    <span class="search-result-score">Relevance: ${Math.round(result.score)}%</span>
                </div>
            </div>
        `;
    },

    // Get type icon
    getTypeIcon: function(type) {
        const icons = {
            section: 'fas fa-folder',
            module: 'fas fa-file-alt',
            topic: 'fas fa-bookmark',
            notebook: 'fas fa-jupyter',
            exercise: 'fas fa-code'
        };
        return icons[type] || 'fas fa-file';
    },

    // Show search results
    showSearchResults: function() {
        const searchResults = document.getElementById('searchResults');
        if (searchResults) {
            searchResults.classList.add('show');
        }
    },

    // Hide search results
    hideSearchResults: function() {
        const searchResults = document.getElementById('searchResults');
        if (searchResults) {
            searchResults.classList.remove('show');
        }
    },

    // Select search result
    selectSearchResult: function(contentId) {
        this.hideSearchResults();
        this.clearSearch();

        // Navigate to the selected content
        const [sectionId, moduleId] = contentId.split('/');

        if (moduleId) {
            AIDocsReader.App.loadModule(sectionId, moduleId);
        } else {
            AIDocsReader.App.loadSection(sectionId);
        }
    },

    // Handle search submit
    handleSearchSubmit: function(query) {
        if (!query || query.length < 2) return;

        const results = this.searchInIndex(query);
        if (results.length > 0) {
            this.selectSearchResult(results[0].id);
        }
    },

    // Clear search
    clearSearch: function() {
        const searchInput = document.getElementById('globalSearch');
        if (searchInput) {
            searchInput.value = '';
        }
        this.hideSearchResults();
    },

    // Advanced search
    advancedSearch: function(params) {
        const {
            query,
            type,
            difficulty,
            section,
            tags,
            dateRange
        } = params;

        let results = this.searchInIndex(query || '');

        // Filter by type
        if (type) {
            results = results.filter(result => result.type === type);
        }

        // Filter by difficulty
        if (difficulty) {
            results = results.filter(result => result.difficulty === difficulty);
        }

        // Filter by section
        if (section) {
            results = results.filter(result => result.section === section);
        }

        // Filter by tags
        if (tags && tags.length > 0) {
            results = results.filter(result =>
                tags.some(tag => result.tags && result.tags.includes(tag))
            );
        }

        // Filter by date range (if we had date information)
        if (dateRange) {
            // Implementation would depend on having date data
        }

        return results;
    },

    // Get search suggestions
    getSearchSuggestions: function(query) {
        if (!query || query.length < 2) return [];

        const suggestions = [];
        const normalizedQuery = query.toLowerCase();

        // Section suggestions
        AIDocsReader.Config.sections.forEach(section => {
            if (section.title.toLowerCase().includes(normalizedQuery)) {
                suggestions.push({
                    type: 'section',
                    text: section.title,
                    description: section.description,
                    action: () => this.selectSearchResult(section.id)
                });
            }
        });

        // Module suggestions (mock implementation)
        const moduleNames = [
            'Theory Foundations', 'Practical Implementations', 'Case Studies',
            'Advanced Topics', 'Exercises and Projects'
        ];

        moduleNames.forEach(module => {
            if (module.toLowerCase().includes(normalizedQuery)) {
                suggestions.push({
                    type: 'module',
                    text: module,
                    description: 'Learn about this topic',
                    action: () => this.handleSearchSubmit(module)
                });
            }
        });

        return suggestions.slice(0, 5);
    },

    // Display search suggestions
    displaySearchSuggestions: function(query) {
        const suggestions = this.getSearchSuggestions(query);
        const searchResults = document.getElementById('searchResults');

        if (suggestions.length === 0) return;

        const suggestionsHtml = suggestions.map(suggestion => `
            <div class="search-suggestion-item" onclick="AIDocsReader.Search.executeSuggestion(${suggestion.action})">
                <div class="suggestion-icon">
                    <i class="${this.getTypeIcon(suggestion.type)}"></i>
                </div>
                <div class="suggestion-content">
                    <div class="suggestion-text">${suggestion.text}</div>
                    <div class="suggestion-description">${suggestion.description}</div>
                </div>
            </div>
        `).join('');

        searchResults.innerHTML = `
            <div class="search-suggestions">
                ${suggestionsHtml}
            </div>
        `;

        this.showSearchResults();
    },

    // Execute suggestion action
    executeSuggestion: function(action) {
        this.hideSearchResults();
        this.clearSearch();
        if (typeof action === 'function') {
            action();
        }
    },

    // Search history
    getSearchHistory: function() {
        return JSON.parse(localStorage.getItem('search-history') || '[]');
    },

    addToSearchHistory: function(query) {
        if (!query || query.length < 2) return;

        const history = this.getSearchHistory();
        const filtered = history.filter(item => item !== query);
        filtered.unshift(query);

        // Keep only last 10 searches
        const limited = filtered.slice(0, 10);
        localStorage.setItem('search-history', JSON.stringify(limited));
    },

    clearSearchHistory: function() {
        localStorage.removeItem('search-history');
    },

    // Display search history
    displaySearchHistory: function() {
        const history = this.getSearchHistory();
        const searchResults = document.getElementById('searchResults');

        if (history.length === 0) {
            searchResults.innerHTML = `
                <div class="search-history-empty">
                    <i class="fas fa-history"></i>
                    <p>No search history</p>
                </div>
            `;
            return;
        }

        const historyHtml = history.map(query => `
            <div class="search-history-item" onclick="AIDocsReader.Search.searchFromHistory('${query}')">
                <i class="fas fa-history"></i>
                <span>${query}</span>
            </div>
        `).join('');

        searchResults.innerHTML = `
            <div class="search-history-header">
                <span>Recent searches</span>
                <button class="btn btn-sm btn-secondary" onclick="AIDocsReader.Search.clearSearchHistory()">
                    Clear
                </button>
            </div>
            <div class="search-history-list">
                ${historyHtml}
            </div>
        `;

        this.showSearchResults();
    },

    // Search from history
    searchFromHistory: function(query) {
        const searchInput = document.getElementById('globalSearch');
        if (searchInput) {
            searchInput.value = query;
        }
        this.performSearch(query);
    }
};