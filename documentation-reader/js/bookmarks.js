// Bookmarks Controller
AIDocsReader.Bookmarks = {
    // Initialize bookmarks
    init: function() {
        this.updateBookmarkCount();
        this.setupBookmarkListeners();
    },

    // Setup bookmark listeners
    setupBookmarkListeners: function() {
        // Save note button
        const saveNoteBtn = document.getElementById('saveNoteBtn');
        if (saveNoteBtn) {
            saveNoteBtn.addEventListener('click', () => this.saveBookmarkNote());
        }

        // Cancel note button
        const cancelNoteBtn = document.getElementById('cancelNoteBtn');
        if (cancelNoteBtn) {
            cancelNoteBtn.addEventListener('click', () => this.closeBookmarkNoteEditor());
        }
    },

    // Toggle bookmark for current content
    toggleBookmark: function(contentId, title, section) {
        const isBookmarked = AIDocsReader.Storage.isBookmarked(contentId);

        if (isBookmarked) {
            this.removeBookmark(contentId);
        } else {
            this.addBookmark(contentId, title, section);
        }

        this.updateBookmarkUI(contentId);
    },

    // Add bookmark
    addBookmark: function(contentId, title, section, note = '') {
        const bookmark = {
            contentId,
            title,
            section,
            note,
            createdAt: Date.now()
        };

        const success = AIDocsReader.Storage.addBookmark(bookmark);

        if (success) {
            this.updateBookmarkCount();
            this.showBookmarkNotification('Bookmark added', 'success');
            this.updateBookmarkUI(contentId);
        }

        return success;
    },

    // Remove bookmark
    removeBookmark: function(contentId) {
        const success = AIDocsReader.Storage.removeBookmark(contentId);

        if (success) {
            this.updateBookmarkCount();
            this.showBookmarkNotification('Bookmark removed', 'info');
            this.updateBookmarkUI(contentId);
        }

        return success;
    },

    // Update bookmark UI
    updateBookmarkUI: function(contentId) {
        const isBookmarked = AIDocsReader.Storage.isBookmarked(contentId);
        const bookmarkBtn = document.getElementById('toggleBookmark');

        if (bookmarkBtn) {
            const icon = bookmarkBtn.querySelector('i');
            if (isBookmarked) {
                icon.classList.remove('far');
                icon.classList.add('fas');
                bookmarkBtn.innerHTML = '<i class="fas fa-bookmark"></i> Bookmarked';
            } else {
                icon.classList.remove('fas');
                icon.classList.add('far');
                bookmarkBtn.innerHTML = '<i class="far fa-bookmark"></i> Bookmark';
            }
        }
    },

    // Update bookmark count
    updateBookmarkCount: function() {
        const bookmarks = AIDocsReader.Storage.getBookmarks();
        const bookmarkCount = document.getElementById('bookmarkCount');

        if (bookmarkCount) {
            bookmarkCount.textContent = bookmarks.length;
            bookmarkCount.style.display = bookmarks.length > 0 ? 'flex' : 'none';
        }
    },

    // Show bookmarks modal
    showBookmarksModal: function() {
        const modal = document.getElementById('bookmarksModal');
        const bookmarksList = document.getElementById('bookmarksList');

        // Render bookmarks
        this.renderBookmarks(bookmarksList);

        // Show modal
        modal.classList.add('active');
    },

    // Render bookmarks
    renderBookmarks: function(container) {
        const bookmarks = AIDocsReader.Storage.getBookmarks();

        if (bookmarks.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-bookmark"></i>
                    <h3>No bookmarks yet</h3>
                    <p>Start bookmarking interesting content to see it here.</p>
                </div>
            `;
            return;
        }

        // Sort bookmarks by creation date (newest first)
        const sortedBookmarks = [...bookmarks].sort((a, b) => b.createdAt - a.createdAt);

        const bookmarksHtml = sortedBookmarks.map(bookmark => `
            <div class="bookmark-item" data-bookmark-id="${bookmark.contentId}">
                <div class="bookmark-content">
                    <div class="bookmark-header">
                        <h4 class="bookmark-title">${bookmark.title}</h4>
                        <div class="bookmark-actions">
                            <button class="bookmark-action-btn edit-bookmark" onclick="AIDocsReader.Bookmarks.editBookmark('${bookmark.contentId}')" title="Edit note">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button class="bookmark-action-btn remove-bookmark" onclick="AIDocsReader.Bookmarks.removeBookmark('${bookmark.contentId}')" title="Remove bookmark">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                    <div class="bookmark-meta">
                        <span class="bookmark-section">${bookmark.section}</span>
                        <span class="bookmark-date">${this.formatDate(bookmark.createdAt)}</span>
                    </div>
                    ${bookmark.note ? `
                        <div class="bookmark-note">
                            <p>${bookmark.note}</p>
                        </div>
                    ` : ''}
                </div>
                <div class="bookmark-navigation">
                    <button class="btn btn-primary btn-sm" onclick="AIDocsReader.Bookmarks.navigateToBookmark('${bookmark.contentId}')">
                        <i class="fas fa-arrow-right"></i>
                        Go to Content
                    </button>
                </div>
            </div>
        `).join('');

        container.innerHTML = bookmarksHtml;
    },

    // Edit bookmark note
    editBookmark: function(contentId) {
        const bookmarks = AIDocsReader.Storage.getBookmarks();
        const bookmark = bookmarks.find(b => b.contentId === contentId);

        if (!bookmark) return;

        const note = prompt('Edit your bookmark note:', bookmark.note || '');
        if (note !== null) {
            const updatedBookmark = { ...bookmark, note, updatedAt: Date.now() };
            AIDocsReader.Storage.addBookmark(updatedBookmark); // This will update existing

            // Refresh the modal
            this.showBookmarksModal();
        }
    },

    // Navigate to bookmark
    navigateToBookmark: function(contentId) {
        // Close modal
        const modal = document.getElementById('bookmarksModal');
        modal.classList.remove('active');

        // Navigate to content (implementation depends on content structure)
        const [sectionId, moduleId] = contentId.split('/');

        if (moduleId) {
            AIDocsReader.App.loadModule(sectionId, moduleId);
        } else {
            AIDocsReader.App.loadSection(sectionId);
        }
    },

    // Add note to bookmark
    addNoteToBookmark: function(contentId) {
        const modal = document.getElementById('bookmarksModal');
        modal.classList.remove('active');

        // Open notes modal with context
        AIDocsReader.Notes.openNotesEditorForContent(contentId);
    },

    // Save bookmark note
    saveBookmarkNote: function() {
        const noteEditor = document.getElementById('bookmarkNoteEditor');
        const saveBtn = document.getElementById('saveBookmarkNoteBtn');

        if (!noteEditor || !saveBtn) return;

        const contentId = saveBtn.dataset.contentId;
        const note = noteEditor.value.trim();

        const bookmarks = AIDocsReader.Storage.getBookmarks();
        const bookmark = bookmarks.find(b => b.contentId === contentId);

        if (bookmark) {
            const updatedBookmark = { ...bookmark, note, updatedAt: Date.now() };
            AIDocsReader.Storage.addBookmark(updatedBookmark);

            this.closeBookmarkNoteEditor();
            this.showBookmarksModal(); // Refresh the bookmarks list
        }
    },

    // Close bookmark note editor
    closeBookmarkNoteEditor: function() {
        const editor = document.getElementById('bookmarkNoteEditor');
        const modal = document.getElementById('bookmarkNoteModal');

        if (editor) editor.value = '';
        if (modal) modal.classList.remove('active');
    },

    // Export bookmarks
    exportBookmarks: function() {
        const bookmarks = AIDocsReader.Storage.getBookmarks();

        const exportData = {
            bookmarks,
            exportedAt: new Date().toISOString(),
            version: '1.0'
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `bookmarks-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    // Import bookmarks
    importBookmarks: function(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);

                    if (data.bookmarks && Array.isArray(data.bookmarks)) {
                        // Merge with existing bookmarks
                        const existingBookmarks = AIDocsReader.Storage.getBookmarks();
                        const newBookmarks = data.bookmarks.filter(newBookmark =>
                            !existingBookmarks.some(existing => existing.contentId === newBookmark.contentId)
                        );

                        const mergedBookmarks = [...existingBookmarks, ...newBookmarks];
                        AIDocsReader.Storage.setBookmarks(mergedBookmarks);

                        this.updateBookmarkCount();
                        resolve(newBookmarks.length);
                    } else {
                        reject(new Error('Invalid bookmarks file format'));
                    }
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    },

    // Search bookmarks
    searchBookmarks: function(query) {
        const bookmarks = AIDocsReader.Storage.getBookmarks();
        const lowerQuery = query.toLowerCase();

        return bookmarks.filter(bookmark =>
            bookmark.title.toLowerCase().includes(lowerQuery) ||
            bookmark.section.toLowerCase().includes(lowerQuery) ||
            (bookmark.note && bookmark.note.toLowerCase().includes(lowerQuery))
        );
    },

    // Get bookmark statistics
    getBookmarkStats: function() {
        const bookmarks = AIDocsReader.Storage.getBookmarks();

        const stats = {
            total: bookmarks.length,
            withNotes: bookmarks.filter(b => b.note && b.note.trim()).length,
            bySection: {},
            recent: bookmarks.filter(b => Date.now() - b.createdAt < 7 * 24 * 60 * 60 * 1000).length
        };

        // Count by section
        bookmarks.forEach(bookmark => {
            stats.bySection[bookmark.section] = (stats.bySection[bookmark.section] || 0) + 1;
        });

        return stats;
    },

    // Format date
    formatDate: function(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        if (diffDays === 1) {
            return 'Yesterday';
        } else if (diffDays < 7) {
            return date.toLocaleDateString('en', { weekday: 'short' });
        } else if (diffDays < 30) {
            return date.toLocaleDateString('en', { month: 'short', day: 'numeric' });
        } else {
            return date.toLocaleDateString('en', { year: 'numeric', month: 'short', day: 'numeric' });
        }
    },

    // Show bookmark notification
    showBookmarkNotification: function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `bookmark-notification bookmark-notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;

        document.body.appendChild(notification);

        setTimeout(() => notification.classList.add('show'), 100);

        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => document.body.removeChild(notification), 300);
        }, 2000);
    }
};