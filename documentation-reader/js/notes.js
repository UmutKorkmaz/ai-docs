// Notes Controller
AIDocsReader.Notes = {
    // Initialize notes
    init: function() {
        this.setupNoteListeners();
        this.updateNotesCount();
    },

    // Setup note listeners
    setupNoteListeners: function() {
        // Save note button
        const saveNoteBtn = document.getElementById('saveNoteBtn');
        if (saveNoteBtn) {
            saveNoteBtn.addEventListener('click', () => this.saveNote());
        }

        // Cancel note button
        const cancelNoteBtn = document.getElementById('cancelNoteBtn');
        if (cancelNoteBtn) {
            cancelNoteBtn.addEventListener('click', () => this.closeNotesEditor());
        }

        // Note editor auto-save
        const noteEditor = document.getElementById('noteEditor');
        if (noteEditor) {
            let autoSaveTimeout;
            noteEditor.addEventListener('input', AIDocsReader.Utils.debounce(() => {
                this.autoSaveNote();
            }, 2000));
        }
    },

    // Show notes modal
    showNotesModal: function() {
        const modal = document.getElementById('notesModal');
        this.renderNotesList();
        modal.classList.add('active');
    },

    // Render notes list
    renderNotesList: function() {
        const notesHistory = document.getElementById('notesHistory');
        const notes = AIDocsReader.Storage.getNotes();

        if (notes.length === 0) {
            notesHistory.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-sticky-note"></i>
                    <h3>No notes yet</h3>
                    <p>Start taking notes while reading to see them here.</p>
                </div>
            `;
            return;
        }

        // Sort notes by creation date (newest first)
        const sortedNotes = [...notes].sort((a, b) => b.createdAt - a.createdAt);

        const notesHtml = sortedNotes.map(note => `
            <div class="note-item" data-note-id="${note.id}">
                <div class="note-header">
                    <div class="note-meta">
                        <span class="note-content-title">${note.contentTitle || 'Unknown Content'}</span>
                        <span class="note-date">${this.formatDate(note.createdAt)}</span>
                    </div>
                    <div class="note-actions">
                        <button class="note-action-btn edit-note" onclick="AIDocsReader.Notes.editNote('${note.id}')" title="Edit note">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="note-action-btn delete-note" onclick="AIDocsReader.Notes.deleteNote('${note.id}')" title="Delete note">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="note-content">
                    <p>${this.truncateText(note.text, 200)}</p>
                </div>
                ${note.tags && note.tags.length > 0 ? `
                    <div class="note-tags">
                        ${note.tags.map(tag => `<span class="note-tag">${tag}</span>`).join('')}
                    </div>
                ` : ''}
            </div>
        `).join('');

        notesHistory.innerHTML = notesHtml;
    },

    // Open notes editor
    openNotesEditor: function(contentId = null, contentTitle = null) {
        const modal = document.getElementById('notesModal');
        const noteEditor = document.getElementById('noteEditor');

        // Store context
        noteEditor.dataset.contentId = contentId || this.getCurrentContentId();
        noteEditor.dataset.contentTitle = contentTitle || this.getCurrentContentTitle();

        // Clear editor
        noteEditor.value = '';

        // Focus editor
        setTimeout(() => noteEditor.focus(), 100);

        // Show modal
        modal.classList.add('active');
    },

    // Open notes editor for specific content
    openNotesEditorForContent: function(contentId) {
        const contentTitle = this.getContentTitle(contentId);
        this.openNotesEditor(contentId, contentTitle);
    },

    // Save note
    saveNote: function() {
        const noteEditor = document.getElementById('noteEditor');
        const noteText = noteEditor.value.trim();

        if (!noteText) {
            this.showNoteNotification('Please enter a note', 'warning');
            return;
        }

        const note = {
            contentId: noteEditor.dataset.contentId || this.getCurrentContentId(),
            contentTitle: noteEditor.dataset.contentTitle || this.getCurrentContentTitle(),
            text: noteText,
            tags: this.extractTags(noteText),
            section: this.getCurrentSection()
        };

        const success = AIDocsReader.Storage.addNote(note);

        if (success) {
            this.showNoteNotification('Note saved successfully', 'success');
            this.closeNotesEditor();
            this.updateNotesCount();
            this.renderNotesList();
            this.updateNotesPanel();

            // Check achievements
            AIDocsReader.Progress.checkAchievements();
        }
    },

    // Auto-save note
    autoSaveNote: function() {
        const noteEditor = document.getElementById('noteEditor');
        const noteText = noteEditor.value.trim();

        if (noteText && noteEditor.dataset.contentId) {
            // Save as draft (implementation could add a 'draft' status)
            const draft = {
                contentId: noteEditor.dataset.contentId,
                contentTitle: noteEditor.dataset.contentTitle || this.getCurrentContentTitle(),
                text: noteText,
                tags: this.extractTags(noteText),
                section: this.getCurrentSection(),
                isDraft: true
            };

            // Store draft in sessionStorage
            sessionStorage.setItem('note-draft', JSON.stringify(draft));
        }
    },

    // Edit note
    editNote: function(noteId) {
        const notes = AIDocsReader.Storage.getNotes();
        const note = notes.find(n => n.id === noteId);

        if (!note) return;

        const noteEditor = document.getElementById('noteEditor');
        noteEditor.value = note.text;
        noteEditor.dataset.noteId = noteId;
        noteEditor.dataset.contentId = note.contentId;
        noteEditor.dataset.contentTitle = note.contentTitle;

        // Focus editor
        setTimeout(() => noteEditor.focus(), 100);

        // Change save button text
        const saveBtn = document.getElementById('saveNoteBtn');
        if (saveBtn) {
            saveBtn.textContent = 'Update Note';
        }
    },

    // Delete note
    deleteNote: function(noteId) {
        if (confirm('Are you sure you want to delete this note?')) {
            const success = AIDocsReader.Storage.deleteNote(noteId);

            if (success) {
                this.showNoteNotification('Note deleted', 'info');
                this.renderNotesList();
                this.updateNotesCount();
                this.updateNotesPanel();
            }
        }
    },

    // Update existing note
    updateNote: function(noteId, updates) {
        const success = AIDocsReader.Storage.updateNote(noteId, updates);

        if (success) {
            this.showNoteNotification('Note updated', 'success');
            this.renderNotesList();
            this.updateNotesPanel();
        }

        return success;
    },

    // Get notes for current content
    getCurrentContentNotes: function() {
        const contentId = this.getCurrentContentId();
        return AIDocsReader.Storage.getNotesForContent(contentId);
    },

    // Update notes panel
    updateNotesPanel: function() {
        const notesList = document.getElementById('notesList');
        if (!notesList) return;

        const notes = this.getCurrentContentNotes();

        if (notes.length === 0) {
            notesList.innerHTML = `
                <div class="no-notes">
                    <p>No notes for this content yet</p>
                    <button class="btn btn-sm btn-primary" onclick="AIDocsReader.Notes.openNotesEditor()">
                        <i class="fas fa-plus"></i> Add Note
                    </button>
                </div>
            `;
            return;
        }

        const notesHtml = notes.slice(0, 3).map(note => `
            <div class="note-item">
                <div class="note-text">${this.truncateText(note.text, 100)}</div>
                <div class="note-meta">
                    <span class="note-date">${this.formatDate(note.createdAt)}</span>
                </div>
            </div>
        `).join('');

        notesList.innerHTML = notesHtml;

        if (notes.length > 3) {
            notesList.innerHTML += `
                <button class="btn btn-sm btn-secondary" onclick="AIDocsReader.Notes.showNotesModal()">
                    View all ${notes.length} notes
                </button>
            `;
        }
    },

    // Close notes editor
    closeNotesEditor: function() {
        const modal = document.getElementById('notesModal');
        const noteEditor = document.getElementById('noteEditor');

        // Clear editor
        noteEditor.value = '';
        delete noteEditor.dataset.noteId;

        // Reset save button
        const saveBtn = document.getElementById('saveNoteBtn');
        if (saveBtn) {
            saveBtn.textContent = 'Save Note';
        }

        // Clear draft
        sessionStorage.removeItem('note-draft');

        modal.classList.remove('active');
    },

    // Search notes
    searchNotes: function(query) {
        const notes = AIDocsReader.Storage.getNotes();
        const lowerQuery = query.toLowerCase();

        return notes.filter(note =>
            note.text.toLowerCase().includes(lowerQuery) ||
            (note.contentTitle && note.contentTitle.toLowerCase().includes(lowerQuery)) ||
            (note.tags && note.tags.some(tag => tag.toLowerCase().includes(lowerQuery)))
        );
    },

    // Get notes by tag
    getNotesByTag: function(tag) {
        const notes = AIDocsReader.Storage.getNotes();
        return notes.filter(note =>
            note.tags && note.tags.includes(tag)
        );
    },

    // Get all tags
    getAllTags: function() {
        const notes = AIDocsReader.Storage.getNotes();
        const tags = new Set();

        notes.forEach(note => {
            if (note.tags) {
                note.tags.forEach(tag => tags.add(tag));
            }
        });

        return Array.from(tags).sort();
    },

    // Extract tags from note text
    extractTags: function(text) {
        const tagRegex = /#(\w+)/g;
        const tags = [];
        let match;

        while ((match = tagRegex.exec(text)) !== null) {
            tags.push(match[1]);
        }

        return tags;
    },

    // Export notes
    exportNotes: function() {
        const notes = AIDocsReader.Storage.getNotes();

        const exportData = {
            notes,
            exportedAt: new Date().toISOString(),
            version: '1.0'
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `notes-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    // Export notes as markdown
    exportNotesAsMarkdown: function() {
        const notes = AIDocsReader.Storage.getNotes();
        let markdown = '# AI Documentation Notes\n\n';
        markdown += `Exported on ${new Date().toLocaleDateString()}\n\n`;

        // Group notes by section
        const notesBySection = {};
        notes.forEach(note => {
            const section = note.section || 'General';
            if (!notesBySection[section]) {
                notesBySection[section] = [];
            }
            notesBySection[section].push(note);
        });

        Object.entries(notesBySection).forEach(([section, sectionNotes]) => {
            markdown += `## ${section}\n\n`;

            sectionNotes.forEach(note => {
                markdown += `### ${note.contentTitle || 'Untitled'}\n\n`;
                markdown += `*Date: ${this.formatDate(note.createdAt)}*\n\n`;
                markdown += `${note.text}\n\n`;

                if (note.tags && note.tags.length > 0) {
                    markdown += `**Tags:** ${note.tags.map(tag => `#${tag}`).join(' ')}\n\n`;
                }

                markdown += '---\n\n';
            });
        });

        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `notes-${new Date().toISOString().split('T')[0]}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    // Get note statistics
    getNoteStats: function() {
        const notes = AIDocsReader.Storage.getNotes();

        const stats = {
            total: notes.length,
            thisWeek: notes.filter(note => Date.now() - note.createdAt < 7 * 24 * 60 * 60 * 1000).length,
            withTags: notes.filter(note => note.tags && note.tags.length > 0).length,
            bySection: {},
            totalWords: notes.reduce((total, note) => total + note.text.split(/\s+/).length, 0)
        };

        // Count by section
        notes.forEach(note => {
            const section = note.section || 'General';
            stats.bySection[section] = (stats.bySection[section] || 0) + 1;
        });

        return stats;
    },

    // Helper methods
    getCurrentContentId: function() {
        const state = AIDocsReader.App.state;
        if (state.currentModule && state.currentSection) {
            return `${state.currentSection}/${state.currentModule}`;
        } else if (state.currentSection) {
            return state.currentSection;
        }
        return null;
    },

    getCurrentContentTitle: function() {
        const state = AIDocsReader.App.state;
        if (state.currentModule) {
            return AIDocsReader.App.formatModuleTitle(state.currentModule);
        } else if (state.currentSection) {
            const section = AIDocsReader.Utils.getSection(state.currentSection);
            return section ? section.title : 'Unknown Section';
        }
        return null;
    },

    getCurrentSection: function() {
        return AIDocsReader.App.state.currentSection;
    },

    getContentTitle: function(contentId) {
        const [sectionId, moduleId] = contentId.split('/');

        if (moduleId) {
            return AIDocsReader.App.formatModuleTitle(moduleId);
        } else {
            const section = AIDocsReader.Utils.getSection(sectionId);
            return section ? section.title : 'Unknown Section';
        }
    },

    updateNotesCount: function() {
        const notes = AIDocsReader.Storage.getNotes();
        const notesBtn = document.getElementById('notesBtn');

        if (notesBtn) {
            const badge = notesBtn.querySelector('.badge');
            if (badge) {
                badge.textContent = notes.length;
                badge.style.display = notes.length > 0 ? 'flex' : 'none';
            }
        }
    },

    truncateText: function(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    },

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

    showNoteNotification: function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `note-notification note-notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
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