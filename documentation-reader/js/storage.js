// Local Storage Management
AIDocsReader.Storage = {
    // Initialize storage
    init: function() {
        // Set default settings if not exists
        if (!this.getSettings()) {
            this.setSettings(AIDocsReader.Config.defaultSettings);
        }

        // Initialize storage structure
        if (!this.getBookmarks()) {
            this.setBookmarks([]);
        }
        if (!this.getProgress()) {
            this.setProgress({});
        }
        if (!this.getNotes()) {
            this.setNotes([]);
        }
        if (!this.getReadingHistory()) {
            this.setReadingHistory([]);
        }
    },

    // Generic storage methods
    set: function(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.error('Storage error:', e);
            return false;
        }
    },

    get: function(key, defaultValue = null) {
        try {
            const value = localStorage.getItem(key);
            return value ? JSON.parse(value) : defaultValue;
        } catch (e) {
            console.error('Storage error:', e);
            return defaultValue;
        }
    },

    remove: function(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            console.error('Storage error:', e);
            return false;
        }
    },

    // Settings management
    setSettings: function(settings) {
        return this.set(AIDocsReader.Config.storageKeys.settings, settings);
    },

    getSettings: function() {
        return this.get(AIDocsReader.Config.storageKeys.settings);
    },

    updateSetting: function(key, value) {
        const settings = this.getSettings() || {};
        settings[key] = value;
        return this.setSettings(settings);
    },

    // Bookmarks management
    setBookmarks: function(bookmarks) {
        return this.set(AIDocsReader.Config.storageKeys.bookmarks, bookmarks);
    },

    getBookmarks: function() {
        return this.get(AIDocsReader.Config.storageKeys.bookmarks, []);
    },

    addBookmark: function(bookmark) {
        const bookmarks = this.getBookmarks();
        const existingIndex = bookmarks.findIndex(b => b.contentId === bookmark.contentId);

        if (existingIndex >= 0) {
            // Update existing bookmark
            bookmarks[existingIndex] = { ...bookmarks[existingIndex], ...bookmark, updatedAt: Date.now() };
        } else {
            // Add new bookmark
            bookmarks.push({
                id: AIDocsReader.Utils.generateId(),
                ...bookmark,
                createdAt: Date.now(),
                updatedAt: Date.now()
            });
        }

        return this.setBookmarks(bookmarks);
    },

    removeBookmark: function(contentId) {
        const bookmarks = this.getBookmarks();
        const filtered = bookmarks.filter(b => b.contentId !== contentId);
        return this.setBookmarks(filtered);
    },

    isBookmarked: function(contentId) {
        const bookmarks = this.getBookmarks();
        return bookmarks.some(b => b.contentId === contentId);
    },

    // Progress management
    setProgress: function(progress) {
        return this.set(AIDocsReader.Config.storageKeys.progress, progress);
    },

    getProgress: function() {
        return this.get(AIDocsReader.Config.storageKeys.progress, {});
    },

    updateContentProgress: function(contentId, progressData) {
        const progress = this.getProgress();
        progress[contentId] = {
            ...progress[contentId],
            ...progressData,
            lastUpdated: Date.now()
        };
        return this.setProgress(progress);
    },

    getContentProgress: function(contentId) {
        const progress = this.getProgress();
        return progress[contentId] || null;
    },

    // Notes management
    setNotes: function(notes) {
        return this.set(AIDocsReader.Config.storageKeys.notes, notes);
    },

    getNotes: function() {
        return this.get(AIDocsReader.Config.storageKeys.notes, []);
    },

    addNote: function(note) {
        const notes = this.getNotes();
        notes.push({
            id: AIDocsReader.Utils.generateId(),
            ...note,
            createdAt: Date.now(),
            updatedAt: Date.now()
        });
        return this.setNotes(notes);
    },

    updateNote: function(noteId, updates) {
        const notes = this.getNotes();
        const index = notes.findIndex(n => n.id === noteId);
        if (index >= 0) {
            notes[index] = { ...notes[index], ...updates, updatedAt: Date.now() };
            return this.setNotes(notes);
        }
        return false;
    },

    deleteNote: function(noteId) {
        const notes = this.getNotes();
        const filtered = notes.filter(n => n.id !== noteId);
        return this.setNotes(filtered);
    },

    getNotesForContent: function(contentId) {
        const notes = this.getNotes();
        return notes.filter(n => n.contentId === contentId);
    },

    // Reading history
    setReadingHistory: function(history) {
        return this.set(AIDocsReader.Config.storageKeys.readingHistory, history);
    },

    getReadingHistory: function() {
        return this.get(AIDocsReader.Config.storageKeys.readingHistory, []);
    },

    addToHistory: function(contentId, title, section) {
        const history = this.getReadingHistory();

        // Remove if already exists
        const filtered = history.filter(h => h.contentId !== contentId);

        // Add to beginning
        filtered.unshift({
            contentId,
            title,
            section,
            timestamp: Date.now()
        });

        // Keep only last 50 items
        const limited = filtered.slice(0, 50);

        return this.setReadingHistory(limited);
    },

    // Last position
    setLastPosition: function(position) {
        return this.set(AIDocsReader.Config.storageKeys.lastPosition, position);
    },

    getLastPosition: function() {
        return this.get(AIDocsReader.Config.storageKeys.lastPosition);
    },

    // Statistics and analytics
    getReadingStats: function() {
        const progress = this.getProgress();
        const history = this.getReadingHistory();
        const notes = this.getNotes();
        const bookmarks = this.getBookmarks();

        // Calculate total reading time
        let totalTime = 0;
        let topicsRead = 0;
        let sectionsCompleted = {};

        Object.values(progress).forEach(p => {
            if (p.timeSpent) totalTime += p.timeSpent;
            if (p.percentage >= 100) {
                topicsRead++;
                if (p.section) {
                    sectionsCompleted[p.section] = (sectionsCompleted[p.section] || 0) + 1;
                }
            }
        });

        // Calculate streaks
        const today = new Date().toDateString();
        const lastReadDate = history.length > 0 ? new Date(history[0].timestamp).toDateString() : null;
        const streak = lastReadDate === today ? 1 : 0;

        return {
            totalTime,
            topicsRead,
            sectionsCompleted: Object.keys(sectionsCompleted).length,
            totalNotes: notes.length,
            totalBookmarks: bookmarks.length,
            streak,
            lastReadDate
        };
    },

    // Export/Import functionality
    exportData: function() {
        const data = {
            settings: this.getSettings(),
            bookmarks: this.getBookmarks(),
            progress: this.getProgress(),
            notes: this.getNotes(),
            history: this.getReadingHistory(),
            exportedAt: Date.now(),
            version: '1.0'
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ai-docs-backup-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    importData: function(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);

                    // Validate data structure
                    if (data.version && data.settings) {
                        // Import data
                        if (data.settings) this.setSettings(data.settings);
                        if (data.bookmarks) this.setBookmarks(data.bookmarks);
                        if (data.progress) this.setProgress(data.progress);
                        if (data.notes) this.setNotes(data.notes);
                        if (data.history) this.setReadingHistory(data.history);

                        resolve(true);
                    } else {
                        reject(new Error('Invalid backup file format'));
                    }
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    },

    // Clear all data
    clearAllData: function() {
        Object.values(AIDocsReader.Config.storageKeys).forEach(key => {
            this.remove(key);
        });
        this.init(); // Reinitialize with defaults
    },

    // Storage usage information
    getStorageInfo: function() {
        let totalSize = 0;
        let usedSize = 0;

        Object.values(AIDocsReader.Config.storageKeys).forEach(key => {
            const value = localStorage.getItem(key);
            if (value) {
                usedSize += value.length;
            }
        });

        totalSize = 5 * 1024 * 1024; // 5MB typical localStorage limit

        return {
            used: usedSize,
            total: totalSize,
            percentage: Math.round((usedSize / totalSize) * 100),
            available: totalSize - usedSize
        };
    }
};