// Navigation Controller
AIDocsReader.Navigation = {
    // Initialize navigation
    init: function() {
        this.setupBreadcrumbNavigation();
        this.setupKeyboardNavigation();
        this.setupTouchGestures();
        this.setupScrollSpy();
    },

    // Setup breadcrumb navigation
    setupBreadcrumbNavigation: function() {
        const breadcrumb = document.getElementById('breadcrumb');

        breadcrumb.addEventListener('click', (e) => {
            if (e.target.classList.contains('breadcrumb-item') && e.target.dataset.section) {
                e.preventDefault();
                const section = e.target.dataset.section;

                if (section === 'home') {
                    AIDocsReader.App.showWelcomeScreen();
                }
            }
        });
    },

    // Setup keyboard navigation
    setupKeyboardNavigation: function() {
        document.addEventListener('keydown', (e) => {
            // Handle arrow key navigation
            if (e.key === 'ArrowLeft' && e.ctrlKey) {
                e.preventDefault();
                this.navigatePrevious();
            } else if (e.key === 'ArrowRight' && e.ctrlKey) {
                e.preventDefault();
                this.navigateNext();
            }
        });
    },

    // Setup touch gestures for mobile
    setupTouchGestures: function() {
        let touchStartX = 0;
        let touchEndX = 0;

        document.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
        });

        document.addEventListener('touchend', (e) => {
            touchEndX = e.changedTouches[0].screenX;
            this.handleSwipe();
        });

        this.handleSwipe = () => {
            const swipeThreshold = 50;
            const diff = touchStartX - touchEndX;

            if (Math.abs(diff) > swipeThreshold) {
                if (diff > 0) {
                    // Swipe left - navigate next
                    this.navigateNext();
                } else {
                    // Swipe right - navigate previous
                    this.navigatePrevious();
                }
            }
        };
    },

    // Setup scroll spy for table of contents
    setupScrollSpy: function() {
        const contentDisplay = document.getElementById('markdownContent');

        if (!contentDisplay) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.updateActiveTOCItem(entry.target.id);
                }
            });
        }, {
            rootMargin: '-20% 0px -70% 0px'
        });

        // Observe all headers
        const observeHeaders = () => {
            contentDisplay.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach(header => {
                if (header.id) {
                    observer.observe(header);
                }
            });
        };

        // Initial observation
        observeHeaders();

        // Re-observe when content changes
        const mutationObserver = new MutationObserver(() => {
            observeHeaders();
        });

        mutationObserver.observe(contentDisplay, {
            childList: true,
            subtree: true
        });
    },

    // Update active table of contents item
    updateActiveTOCItem: function(activeId) {
        document.querySelectorAll('.toc-item').forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === `#${activeId}`) {
                item.classList.add('active');
            }
        });
    },

    // Navigate to previous content
    navigatePrevious: function() {
        const history = AIDocsReader.Storage.getReadingHistory();
        if (history.length > 1) {
            const previousItem = history[1];
            this.navigateToContent(previousItem.contentId, previousItem.section);
        }
    },

    // Navigate to next content
    navigateNext: function() {
        // This would need content structure to determine next item
        console.log('Navigate to next content');
    },

    // Navigate to specific content
    navigateToContent: function(contentId, section) {
        // Implementation depends on content structure
        console.log('Navigate to content:', contentId, section);
    },

    // Generate navigation trail
    generateNavigationTrail: function(currentSection, currentModule, currentContent) {
        const trail = [];

        trail.push({
            title: 'Home',
            action: () => AIDocsReader.App.showWelcomeScreen()
        });

        if (currentSection) {
            const section = AIDocsReader.Utils.getSection(currentSection);
            if (section) {
                trail.push({
                    title: section.title,
                    action: () => AIDocsReader.App.loadSection(currentSection)
                });
            }
        }

        if (currentModule) {
            trail.push({
                title: AIDocsReader.App.formatModuleTitle(currentModule),
                action: () => AIDocsReader.App.loadModule(currentSection, currentModule)
            });
        }

        if (currentContent) {
            trail.push({
                title: currentContent,
                action: null
            });
        }

        return trail;
    },

    // Update navigation UI
    updateNavigationUI: function(trail) {
        const breadcrumb = document.getElementById('breadcrumb');

        const breadcrumbHtml = trail.map((item, index) => {
            if (index === trail.length - 1) {
                return `<span class="breadcrumb-item current">${item.title}</span>`;
            } else {
                return `<a href="#" class="breadcrumb-item" onclick="AIDocsReader.Navigation.navigateToTrailItem(${index})">${item.title}</a>`;
            }
        }).join('');

        breadcrumb.innerHTML = breadcrumbHtml;
    },

    // Navigate to trail item
    navigateToTrailItem: function(index) {
        const trail = this.getCurrentTrail();
        if (trail && trail[index] && trail[index].action) {
            trail[index].action();
        }
    },

    // Get current navigation trail
    getCurrentTrail: function() {
        // This should return the current trail based on app state
        return AIDocsReader.App.state.currentTrail || [];
    }
};