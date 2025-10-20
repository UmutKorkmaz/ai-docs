// Theme Controller
AIDocsReader.Themes = {
    // Initialize themes
    init: function() {
        this.setupThemeListeners();
        this.loadThemePreferences();
        this.applyTheme();
    },

    // Setup theme listeners
    setupThemeListeners: function() {
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.cycleTheme());
        }

        // Listen for system theme changes
        if (window.matchMedia) {
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            darkModeQuery.addEventListener('change', (e) => {
                if (this.shouldUseSystemTheme()) {
                    this.setSystemTheme(e.matches ? 'dark' : 'light');
                }
            });
        }
    },

    // Load theme preferences
    loadThemePreferences: function() {
        const settings = AIDocsReader.Storage.getSettings();
        this.currentTheme = settings.theme || 'light';
        this.systemThemeEnabled = settings.useSystemTheme || false;
    },

    // Set theme
    setTheme: function(themeName) {
        if (!this.isValidTheme(themeName)) {
            console.warn(`Invalid theme: ${themeName}`);
            return;
        }

        this.currentTheme = themeName;
        document.documentElement.setAttribute('data-theme', themeName);
        this.updateThemeUI(themeName);
        AIDocsReader.Storage.updateSetting('theme', themeName);

        // Apply theme-specific customizations
        this.applyThemeCustomizations(themeName);
    },

    // Cycle through themes
    cycleTheme: function() {
        const themes = ['light', 'dark', 'high-contrast', 'sepia', 'eye-care'];
        const currentIndex = themes.indexOf(this.currentTheme);
        const nextIndex = (currentIndex + 1) % themes.length;
        this.setTheme(themes[nextIndex]);
    },

    // Toggle light/dark theme
    toggleLightDark: function() {
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    },

    // Apply theme customizations
    applyThemeCustomizations: function(themeName) {
        const settings = AIDocsReader.Storage.getSettings();

        // Apply font settings
        if (settings.fontSize) {
            document.documentElement.style.fontSize = `${settings.fontSize}px`;
        }

        // Apply font family
        if (settings.fontFamily) {
            document.body.style.fontFamily = settings.fontFamily;
        }

        // Apply line height
        if (settings.lineHeight) {
            document.documentElement.style.setProperty('--line-height', settings.lineHeight);
        }

        // Apply max width
        if (settings.maxWidth) {
            document.documentElement.style.setProperty('--content-max-width', `${settings.maxWidth}px`);
        }

        // Theme-specific customizations
        switch (themeName) {
            case 'high-contrast':
                this.applyHighContrastSettings();
                break;
            case 'eye-care':
                this.applyEyeCareSettings();
                break;
            case 'sepia':
                this.applySepiaSettings();
                break;
        }
    },

    // Update theme UI
    updateThemeUI: function(themeName) {
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            const icon = themeToggle.querySelector('i');
            const icons = {
                'light': 'fa-moon',
                'dark': 'fa-sun',
                'high-contrast': 'fa-adjust',
                'sepia': 'fa-coffee',
                'eye-care': 'fa-eye'
            };

            icon.className = `fas ${icons[themeName] || 'fa-moon'}`;
            themeToggle.title = `Switch to ${this.getNextThemeName()} theme`;
        }
    },

    // Get next theme name
    getNextThemeName: function() {
        const themes = ['light', 'dark', 'high-contrast', 'sepia', 'eye-care'];
        const currentIndex = themes.indexOf(this.currentTheme);
        const nextIndex = (currentIndex + 1) % themes.length;
        return themes[nextIndex];
    },

    // Validate theme name
    isValidTheme: function(themeName) {
        const validThemes = ['light', 'dark', 'high-contrast', 'sepia', 'eye-care'];
        return validThemes.includes(themeName);
    },

    // Apply high contrast settings
    applyHighContrastSettings: function() {
        // Increase font sizes and improve contrast
        document.documentElement.style.setProperty('--font-size-multiplier', '1.1');

        // Ensure high contrast for text elements
        const style = document.createElement('style');
        style.id = 'high-contrast-styles';
        style.textContent = `
            .markdown-content h1, .markdown-content h2, .markdown-content h3,
            .markdown-content h4, .markdown-content h5, .markdown-content h6 {
                border-bottom: 2px solid currentColor;
            }
            .code-block {
                border: 2px solid var(--border);
            }
            a {
                text-decoration: underline;
            }
            .btn {
                border: 2px solid currentColor;
            }
        `;

        // Remove existing styles
        const existingStyle = document.getElementById('high-contrast-styles');
        if (existingStyle) {
            existingStyle.remove();
        }

        document.head.appendChild(style);
    },

    // Apply eye care settings
    applyEyeCareSettings: function() {
        // Warmer color temperature and reduced blue light
        const style = document.createElement('style');
        style.id = 'eye-care-styles';
        style.textContent = `
            body {
                filter: sepia(20%) saturate(90%);
            }
            img {
                filter: sepia(10%) saturate(80%);
            }
            .markdown-content {
                color: #2c3e50;
            }
            .markdown-content a {
                color: #27ae60;
            }
            .markdown-content code {
                background-color: #f0f0e6;
                color: #2c3e50;
            }
        `;

        // Remove existing styles
        const existingStyle = document.getElementById('eye-care-styles');
        if (existingStyle) {
            existingStyle.remove();
        }

        document.head.appendChild(style);
    },

    // Apply sepia settings
    applySepiaSettings: function() {
        // Classic book-like appearance
        const style = document.createElement('style');
        style.id = 'sepia-styles';
        style.textContent = `
            .markdown-content {
                font-family: 'Georgia', serif;
                color: #5d4037;
                line-height: 1.8;
            }
            .markdown-content h1, .markdown-content h2,
            .markdown-content h3, .markdown-content h4 {
                font-family: 'Georgia', serif;
                font-weight: 600;
            }
            .markdown-content a {
                color: #8d6e63;
            }
            .markdown-content blockquote {
                border-left: 4px solid #8d6e63;
                background-color: rgba(141, 110, 99, 0.1);
            }
            .markdown-content code {
                background-color: #dcd4b8;
                color: #5d4037;
            }
        `;

        // Remove existing styles
        const existingStyle = document.getElementById('sepia-styles');
        if (existingStyle) {
            existingStyle.remove();
        }

        document.head.appendChild(style);
    },

    // Remove theme-specific styles
    removeThemeStyles: function() {
        const styleIds = ['high-contrast-styles', 'eye-care-styles', 'sepia-styles'];
        styleIds.forEach(id => {
            const style = document.getElementById(id);
            if (style) {
                style.remove();
            }
        });
    },

    // Apply theme (initial application)
    applyTheme: function() {
        this.setTheme(this.currentTheme);
    },

    // Check if should use system theme
    shouldUseSystemTheme: function() {
        const settings = AIDocsReader.Storage.getSettings();
        return settings.useSystemTheme || false;
    },

    // Set system theme
    setSystemTheme: function(systemTheme) {
        if (this.shouldUseSystemTheme()) {
            this.setTheme(systemTheme);
        }
    },

    // Get available themes
    getAvailableThemes: function() {
        return [
            {
                name: 'light',
                displayName: 'Light',
                description: 'Default light theme',
                icon: 'fa-sun'
            },
            {
                name: 'dark',
                displayName: 'Dark',
                description: 'Dark theme for night reading',
                icon: 'fa-moon'
            },
            {
                name: 'high-contrast',
                displayName: 'High Contrast',
                description: 'Maximum contrast for accessibility',
                icon: 'fa-adjust'
            },
            {
                name: 'sepia',
                displayName: 'Sepia',
                description: 'Warm, book-like appearance',
                icon: 'fa-coffee'
            },
            {
                name: 'eye-care',
                displayName: 'Eye Care',
                description: 'Reduced blue light for comfortable reading',
                icon: 'fa-eye'
            }
        ];
    },

    // Show theme selector
    showThemeSelector: function() {
        const themes = this.getAvailableThemes();
        const currentTheme = this.currentTheme;

        const themeSelector = document.createElement('div');
        themeSelector.className = 'theme-selector';
        themeSelector.innerHTML = `
            <div class="theme-selector-header">
                <h3>Choose Theme</h3>
                <button class="theme-selector-close" onclick="AIDocsReader.Themes.closeThemeSelector()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="theme-options">
                ${themes.map(theme => `
                    <div class="theme-option ${theme.name === currentTheme ? 'active' : ''}"
                         onclick="AIDocsReader.Themes.setTheme('${theme.name}')">
                        <div class="theme-preview">
                            <i class="fas ${theme.icon}"></i>
                        </div>
                        <div class="theme-info">
                            <div class="theme-name">${theme.displayName}</div>
                            <div class="theme-description">${theme.description}</div>
                        </div>
                        ${theme.name === currentTheme ? '<i class="fas fa-check"></i>' : ''}
                    </div>
                `).join('')}
            </div>
            <div class="theme-settings">
                <label class="theme-setting">
                    <input type="checkbox" id="useSystemTheme" ${this.systemThemeEnabled ? 'checked' : ''}
                           onchange="AIDocsReader.Themes.toggleSystemTheme()">
                    Use system theme preference
                </label>
            </div>
        `;

        document.body.appendChild(themeSelector);

        // Close when clicking outside
        setTimeout(() => {
            document.addEventListener('click', this.closeThemeSelectorHandler);
        }, 100);
    },

    // Close theme selector
    closeThemeSelector: function() {
        const themeSelector = document.querySelector('.theme-selector');
        if (themeSelector) {
            themeSelector.remove();
        }
        document.removeEventListener('click', this.closeThemeSelectorHandler);
    },

    // Close theme selector handler
    closeThemeSelectorHandler: function(e) {
        const themeSelector = document.querySelector('.theme-selector');
        if (themeSelector && !themeSelector.contains(e.target)) {
            AIDocsReader.Themes.closeThemeSelector();
        }
    },

    // Toggle system theme preference
    toggleSystemTheme: function() {
        const checkbox = document.getElementById('useSystemTheme');
        const useSystemTheme = checkbox.checked;

        AIDocsReader.Storage.updateSetting('useSystemTheme', useSystemTheme);
        this.systemThemeEnabled = useSystemTheme;

        if (useSystemTheme && window.matchMedia) {
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            this.setSystemTheme(darkModeQuery.matches ? 'dark' : 'light');
        }
    },

    // Get theme statistics
    getThemeStats: function() {
        const settings = AIDocsReader.Storage.getSettings();
        return {
            currentTheme: this.currentTheme,
            systemThemeEnabled: this.systemThemeEnabled,
            fontSize: settings.fontSize,
            fontFamily: settings.fontFamily,
            lineHeight: settings.lineHeight,
            maxWidth: settings.maxWidth
        };
    },

    // Export theme settings
    exportThemeSettings: function() {
        const settings = AIDocsReader.Storage.getSettings();
        const themeSettings = {
            theme: settings.theme,
            fontSize: settings.fontSize,
            fontFamily: settings.fontFamily,
            lineHeight: settings.lineHeight,
            maxWidth: settings.maxWidth,
            useSystemTheme: settings.useSystemTheme
        };

        return themeSettings;
    },

    // Import theme settings
    importThemeSettings: function(themeSettings) {
        if (themeSettings.theme) {
            this.setTheme(themeSettings.theme);
        }

        Object.keys(themeSettings).forEach(key => {
            if (key !== 'theme') {
                AIDocsReader.Storage.updateSetting(key, themeSettings[key]);
            }
        });

        this.applyThemeCustomizations(this.currentTheme);
    },

    // Reset theme settings to default
    resetThemeSettings: function() {
        const defaultSettings = AIDocsReader.Config.defaultSettings;
        Object.keys(defaultSettings).forEach(key => {
            AIDocsReader.Storage.updateSetting(key, defaultSettings[key]);
        });

        this.setTheme('light');
    }
};