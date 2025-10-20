// Progress Tracking Controller
AIDocsReader.Progress = {
    // Initialize progress tracking
    init: function() {
        this.setupScrollTracking();
        this.setupTimeTracking();
        this.updateProgressDisplay();
        this.checkAchievements();
    },

    // Setup scroll tracking for reading progress
    setupScrollTracking: function() {
        const contentDisplay = document.getElementById('markdownContent');

        if (!contentDisplay) return;

        let scrollTimeout;
        let hasTracked = false;

        contentDisplay.addEventListener('scroll', AIDocsReader.Utils.throttle(() => {
            if (scrollTimeout) clearTimeout(scrollTimeout);

            scrollTimeout = setTimeout(() => {
                const scrollProgress = this.calculateScrollProgress(contentDisplay);

                if (scrollProgress > 0.9 && !hasTracked) {
                    // Mark as completed when scrolled to 90%
                    this.markContentAsCompleted();
                    hasTracked = true;
                } else {
                    // Update progress
                    this.updateContentProgress(Math.round(scrollProgress * 100));
                }
            }, 1000);
        }, 500));
    },

    // Setup time tracking
    setupTimeTracking: function() {
        let startTime = Date.now();
        let isActive = true;

        // Track time on page
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Page hidden, pause tracking
                const timeSpent = Date.now() - startTime;
                this.updateTimeSpent(timeSpent);
                isActive = false;
            } else {
                // Page visible, resume tracking
                startTime = Date.now();
                isActive = true;
            }
        });

        // Track time before page unload
        window.addEventListener('beforeunload', () => {
            if (isActive) {
                const timeSpent = Date.now() - startTime;
                this.updateTimeSpent(timeSpent);
            }
        });

        // Save progress periodically
        setInterval(() => {
            if (isActive) {
                const timeSpent = Date.now() - startTime;
                this.updateTimeSpent(timeSpent);
                startTime = Date.now(); // Reset start time
            }
        }, 30000); // Every 30 seconds
    },

    // Calculate scroll progress
    calculateScrollProgress: function(element) {
        const scrollTop = element.scrollTop;
        const scrollHeight = element.scrollHeight - element.clientHeight;
        return scrollHeight > 0 ? scrollTop / scrollHeight : 0;
    },

    // Update content progress
    updateContentProgress: function(percentage) {
        const currentContent = this.getCurrentContentId();
        if (!currentContent) return;

        const progressData = {
            percentage,
            lastUpdated: Date.now()
        };

        AIDocsReader.Storage.updateContentProgress(currentContent, progressData);
        this.updateProgressDisplay();
    },

    // Mark content as completed
    markContentAsCompleted: function() {
        const currentContent = this.getCurrentContentId();
        if (!currentContent) return;

        const progressData = {
            percentage: 100,
            completed: true,
            completedAt: Date.now(),
            lastUpdated: Date.now()
        };

        AIDocsReader.Storage.updateContentProgress(currentContent, progressData);
        this.updateProgressDisplay();
        this.checkAchievements();
        this.showCompletionNotification();
    },

    // Update time spent
    updateTimeSpent: function(additionalTime) {
        const currentContent = this.getCurrentContentId();
        if (!currentContent) return;

        const existingProgress = AIDocsReader.Storage.getContentProgress(currentContent) || {};
        const currentTimeSpent = existingProgress.timeSpent || 0;

        const progressData = {
            timeSpent: currentTimeSpent + additionalTime,
            lastUpdated: Date.now()
        };

        AIDocsReader.Storage.updateContentProgress(currentContent, progressData);
    },

    // Get current content ID
    getCurrentContentId: function() {
        const state = AIDocsReader.App.state;
        if (state.currentModule && state.currentSection) {
            return `${state.currentSection}/${state.currentModule}`;
        } else if (state.currentSection) {
            return state.currentSection;
        }
        return null;
    },

    // Update progress display
    updateProgressDisplay: function() {
        const stats = AIDocsReader.Storage.getReadingStats();

        // Update overall progress
        const overallProgress = document.getElementById('overallProgress');
        if (overallProgress) {
            const totalTopics = 1500; // Approximate total topics
            const progressPercentage = AIDocsReader.Utils.calculateProgress(stats.topicsRead, totalTopics);

            overallProgress.style.background = `conic-gradient(var(--primary-color) ${progressPercentage * 3.6}deg, var(--border) 0deg)`;
            overallProgress.querySelector('.progress-value').textContent = `${progressPercentage}%`;
        }

        // Update stats
        this.updateStatDisplay('topicsRead', stats.topicsRead);
        this.updateStatDisplay('timeSpent', AIDocsReader.Utils.formatDuration(stats.totalTime / 1000 / 60));

        // Update section progress
        this.updateSectionProgress();
    },

    // Update stat display
    updateStatDisplay: function(statId, value) {
        const element = document.getElementById(statId);
        if (element) {
            element.textContent = value;
        }
    },

    // Update section progress
    updateSectionProgress: function() {
        const state = AIDocsReader.App.state;
        if (!state.currentSection) return;

        const sectionProgress = AIDocsReader.App.getSectionProgress(state.currentSection);
        const progressBar = document.getElementById('sectionProgress');

        if (progressBar) {
            progressBar.style.width = `${sectionProgress}%`;
        }
    },

    // Check achievements
    checkAchievements: function() {
        const stats = AIDocsReader.Storage.getReadingStats();
        const achievements = AIDocsReader.Config.achievements;
        const unlockedAchievements = AIDocsReader.Storage.getSettings().achievements || [];

        Object.entries(achievements).forEach(([key, achievement]) => {
            if (!unlockedAchievements.includes(key) && this.isAchievementUnlocked(key, stats)) {
                this.unlockAchievement(key, achievement);
            }
        });
    },

    // Check if achievement is unlocked
    isAchievementUnlocked: function(achievementKey, stats) {
        switch (achievementKey) {
            case 'first_read':
                return stats.topicsRead >= 1;
            case 'marathon_reader':
                return stats.totalTime >= 3 * 60 * 60 * 1000; // 3 hours
            case 'completionist':
                return stats.sectionsCompleted >= 1;
            case 'note_taker':
                return stats.totalNotes >= 10;
            case 'explorer':
                return stats.sectionsCompleted >= 5; // Visited all main sections
            case 'night_owl':
                return this.hasNightReading();
            default:
                return false;
        }
    },

    // Check for night reading
    hasNightReading: function() {
        const history = AIDocsReader.Storage.getReadingHistory();
        const oneHourInMs = 60 * 60 * 1000;

        return history.some(entry => {
            const entryDate = new Date(entry.timestamp);
            const entryHour = entryDate.getHours();
            return entryHour >= 22 || entryHour <= 6; // Between 10 PM and 6 AM
        });
    },

    // Unlock achievement
    unlockAchievement: function(key, achievement) {
        const settings = AIDocsReader.Storage.getSettings();
        const achievements = settings.achievements || [];

        achievements.push(key);
        AIDocsReader.Storage.updateSetting('achievements', achievements);

        this.showAchievementNotification(achievement);
    },

    // Show achievement notification
    showAchievementNotification: function(achievement) {
        const notification = document.createElement('div');
        notification.className = 'achievement-notification';
        notification.innerHTML = `
            <div class="achievement-content">
                <i class="${achievement.icon}" style="color: ${achievement.color};"></i>
                <div>
                    <div class="achievement-title">Achievement Unlocked!</div>
                    <div class="achievement-name">${achievement.name}</div>
                    <div class="achievement-description">${achievement.description}</div>
                </div>
            </div>
        `;

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);

        // Remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => document.body.removeChild(notification), 300);
        }, 5000);
    },

    // Show completion notification
    showCompletionNotification: function() {
        const notification = document.createElement('div');
        notification.className = 'completion-notification';
        notification.innerHTML = `
            <div class="completion-content">
                <i class="fas fa-check-circle" style="color: var(--success-color);"></i>
                <div>
                    <div class="completion-title">Topic Completed!</div>
                    <div class="completion-message">Great job! You've finished this topic.</div>
                </div>
            </div>
        `;

        document.body.appendChild(notification);

        setTimeout(() => notification.classList.add('show'), 100);

        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => document.body.removeChild(notification), 300);
        }, 3000);
    },

    // Show progress modal
    showProgressModal: function() {
        const modal = document.getElementById('progressModal');
        const stats = AIDocsReader.Storage.getReadingStats();

        // Update modal content
        this.updateProgressModalContent(stats);

        // Show modal
        modal.classList.add('active');

        // Initialize chart if available
        this.initializeProgressChart(stats);
    },

    // Update progress modal content
    updateProgressModalContent: function(stats) {
        // Update overview stats
        const modal = document.getElementById('progressModal');

        // Find or create stats display
        let statsDisplay = modal.querySelector('.progress-stats-display');
        if (!statsDisplay) {
            statsDisplay = document.createElement('div');
            statsDisplay.className = 'progress-stats-display';
            modal.querySelector('.modal-body').prepend(statsDisplay);
        }

        statsDisplay.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-book"></i>
                    </div>
                    <div class="stat-info">
                        <div class="stat-value">${stats.topicsRead}</div>
                        <div class="stat-label">Topics Read</div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-clock"></i>
                    </div>
                    <div class="stat-info">
                        <div class="stat-value">${AIDocsReader.Utils.formatDuration(stats.totalTime / 1000 / 60)}</div>
                        <div class="stat-label">Time Spent</div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-sticky-note"></i>
                    </div>
                    <div class="stat-info">
                        <div class="stat-value">${stats.totalNotes}</div>
                        <div class="stat-label">Notes</div>
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">
                        <i class="fas fa-bookmark"></i>
                    </div>
                    <div class="stat-info">
                        <div class="stat-value">${stats.totalBookmarks}</div>
                        <div class="stat-label">Bookmarks</div>
                    </div>
                </div>
            </div>
        `;

        // Update achievements
        this.updateAchievementsDisplay();
    },

    // Update achievements display
    updateAchievementsDisplay: function() {
        const achievements = AIDocsReader.Config.achievements;
        const unlockedAchievements = AIDocsReader.Storage.getSettings().achievements || [];
        const achievementsList = document.getElementById('achievementsList');

        if (!achievementsList) return;

        const achievementsHtml = Object.entries(achievements).map(([key, achievement]) => {
            const isUnlocked = unlockedAchievements.includes(key);
            return `
                <div class="achievement-item ${isUnlocked ? 'unlocked' : 'locked'}">
                    <div class="achievement-icon">
                        <i class="${achievement.icon}" style="color: ${isUnlocked ? achievement.color : 'var(--text-muted)'};"></i>
                    </div>
                    <div class="achievement-info">
                        <div class="achievement-name">${achievement.name}</div>
                        <div class="achievement-description">${achievement.description}</div>
                    </div>
                    ${isUnlocked ? '<div class="achievement-status"><i class="fas fa-check"></i></div>' : ''}
                </div>
            `;
        }).join('');

        achievementsList.innerHTML = achievementsHtml;
    },

    // Initialize progress chart
    initializeProgressChart: function(stats) {
        const chartCanvas = document.getElementById('progressChart');
        if (!chartCanvas) return;

        // Destroy existing chart if present
        if (this.progressChart) {
            this.progressChart.destroy();
        }

        // Get reading history for the last 7 days
        const history = AIDocsReader.Storage.getReadingHistory();
        const last7Days = this.getLast7DaysData(history);

        this.progressChart = new Chart(chartCanvas, {
            type: 'line',
            data: {
                labels: last7Days.labels,
                datasets: [{
                    label: 'Reading Time (minutes)',
                    data: last7Days.data,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Minutes'
                        }
                    }
                }
            }
        });
    },

    // Get last 7 days data
    getLast7DaysData: function(history) {
        const labels = [];
        const data = [];
        const today = new Date();

        for (let i = 6; i >= 0; i--) {
            const date = new Date(today);
            date.setDate(today.getDate() - i);
            const dateString = date.toLocaleDateString('en', { weekday: 'short' });

            labels.push(dateString);

            // Calculate reading time for this day
            const dayStart = new Date(date.setHours(0, 0, 0, 0));
            const dayEnd = new Date(date.setHours(23, 59, 59, 999));

            const dayHistory = history.filter(entry => {
                const entryDate = new Date(entry.timestamp);
                return entryDate >= dayStart && entryDate <= dayEnd;
            });

            // This is simplified - in reality, you'd need to track time spent per session
            const readingTime = dayHistory.length * 10; // Assume 10 minutes per entry
            data.push(readingTime);
        }

        return { labels, data };
    }
};