// Markdown Content Renderer
AIDocsReader.Renderer = {
    // Initialize marked options
    init: function() {
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                highlight: function(code, lang) {
                    if (Prism && lang && Prism.languages[lang]) {
                        return Prism.highlight(code, Prism.languages[lang], lang);
                    }
                    return code;
                },
                breaks: true,
                gfm: true,
                tables: true,
                sanitize: false,
                smartLists: true,
                smartypants: true
            });
        }
    },

    // Render markdown content
    render: function(markdown) {
        if (!markdown) return '';

        let html = marked ? marked.parse(markdown) : this.fallbackRender(markdown);

        // Process custom features
        html = this.processCustomElements(html);
        html = this.addCodeCopyButtons(html);
        html = this.processImages(html);
        html = this.addAnchors(html);

        return html;
    },

    // Fallback renderer (basic)
    fallbackRender: function(markdown) {
        return markdown
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
            .replace(/\*(.*)\*/gim, '<em>$1</em>')
            .replace(/```(.*?)```/gims, '<pre><code>$1</code></pre>')
            .replace(/`(.*?)`/gim, '<code>$1</code>')
            .replace(/\n\n/gim, '</p><p>')
            .replace(/\n/gim, '<br>')
            .replace(/^(.*)$/gim, '<p>$1</p>');
    },

    // Process custom elements
    processCustomElements: function(html) {
        // Process code blocks with language
        html = html.replace(/<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g,
            '<div class="code-block" data-language="$1"><pre><code>$2</code></pre><button class="copy-code-btn" title="Copy code"><i class="fas fa-copy"></i></button></div>');

        // Process tables
        html = html.replace(/<table>/g, '<div class="table-wrapper"><table class="markdown-table">');
        html = html.replace(/<\/table>/g, '</table></div>');

        // Process blockquotes
        html = html.replace(/<blockquote>/g, '<blockquote class="markdown-quote">');

        // Process task lists
        html = html.replace(/<li>\s*\[ \]/g, '<li class="task-item"><input type="checkbox" disabled>');
        html = html.replace(/<li>\s*\[x\]/g, '<li class="task-item completed"><input type="checkbox" checked disabled>');

        return html;
    },

    // Add copy buttons to code blocks
    addCodeCopyButtons: function(html) {
        // This is handled in processCustomElements for better control
        return html;
    },

    // Process images with lazy loading and captions
    processImages: function(html) {
        html = html.replace(/<img([^>]+)src="([^"]+)"([^>]*)>/g, (match, before, src, after) => {
            const alt = (before + after).match(/alt="([^"]*)"/);
            const altText = alt ? alt[1] : '';

            return `
                <div class="image-container">
                    <img${before}src="${src}"${after} loading="lazy" alt="${altText}">
                    ${altText ? `<div class="image-caption">${altText}</div>` : ''}
                </div>
            `;
        });

        return html;
    },

    // Add anchor links to headers
    addAnchors: function(html) {
        const headerRegex = /<h([1-6])([^>]+)>(.*?)<\/h[1-6]>/g;

        return html.replace(headerRegex, (match, level, attrs, text) => {
            const id = this.generateHeaderId(text);
            return `
                <h${level} id="${id}"${attrs}>
                    <a href="#${id}" class="header-anchor">#</a>
                    ${text}
                </h${level}>
            `;
        });
    },

    // Generate header ID from text
    generateHeaderId: function(text) {
        return text
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
            .replace(/^-|-$/g, '');
    },

    // Extract table of contents
    extractTableOfContents: function(html) {
        const headers = [];
        const headerRegex = /<h([1-6])([^>]*)id="([^"]*)"[^>]*>(.*?)<\/h[1-6]>/g;
        let match;

        while ((match = headerRegex.exec(html)) !== null) {
            const level = parseInt(match[1]);
            const id = match[3];
            const text = match[4].replace(/<[^>]*>/g, '').trim();

            headers.push({
                level,
                id,
                text,
                element: `h${level}`
            });
        }

        return headers;
    },

    // Render table of contents
    renderTableOfContents: function(headers) {
        if (!headers || headers.length === 0) return '';

        const tocHtml = headers.map(header => `
            <a href="#${header.id}" class="toc-item level-${header.level}" data-level="${header.level}">
                ${header.text}
            </a>
        `).join('');

        return tocHtml;
    },

    // Estimate reading time
    estimateReadingTime: function(content) {
        const wordsPerMinute = 200;
        const words = content.trim().split(/\s+/).length;
        const minutes = Math.ceil(words / wordsPerMinute);
        return AIDocsReader.Utils.formatDuration(minutes);
    },

    // Highlight search terms
    highlightSearchTerms: function(content, searchTerm) {
        if (!searchTerm) return content;

        const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return content.replace(regex, '<mark class="search-highlight">$1</mark>');
    },

    // Process mathematical expressions (if MathJax is available)
    processMath: function(html) {
        if (window.MathJax) {
            html = html.replace(/\$\$(.*?)\$\$/g, '\\[ $1 \\]');
            html = html.replace(/\$(.*?)\$/g, '\\( $1 \\)');

            // Typeset after a short delay
            setTimeout(() => {
                if (MathJax.typesetPromise) {
                    MathJax.typesetPromise([document.getElementById('markdownContent')]);
                }
            }, 100);
        }

        return html;
    },

    // Add interactive elements
    addInteractiveElements: function(html) {
        // Add collapsible sections
        html = html.replace(/<details>(.*?)<\/details>/gs, (match, content) => {
            return `<div class="collapsible-section">
                <button class="collapsible-toggle">
                    <i class="fas fa-chevron-right"></i>
                    <span class="collapsible-title">${content.match(/<summary>(.*?)<\/summary>/)?.[1] || 'Details'}</span>
                </button>
                <div class="collapsible-content">${content.replace(/<summary>.*?<\/summary>/, '')}</div>
            </div>`;
        });

        // Add spoiler tags
        html = html.replace(/\|\|(.*?)\|\|/g, '<span class="spoiler" onclick="this.classList.toggle(\'revealed\')">$1</span>');

        return html;
    },

    // Enhance code blocks with line numbers
    addLineNumbers: function(code, language) {
        const lines = code.split('\n');
        const lineNumbers = lines.map((_, i) => i + 1).join('\n');

        return `
            <div class="code-block-with-lines" data-language="${language}">
                <div class="line-numbers">${lineNumbers}</div>
                <pre><code class="language-${language}">${code}</code></pre>
                <button class="copy-code-btn" title="Copy code"><i class="fas fa-copy"></i></button>
            </div>
        `;
    },

    // Process mermaid diagrams
    processMermaidDiagrams: function(html) {
        if (window.mermaid) {
            html = html.replace(/```mermaid\n([\s\S]*?)```/g, (match, diagram) => {
                const id = 'mermaid-' + Math.random().toString(36).substr(2, 9);
                return `<div class="mermaid" id="${id}">${diagram}</div>`;
            });

            // Initialize diagrams after rendering
            setTimeout(() => {
                mermaid.init();
            }, 100);
        }

        return html;
    },

    // Add footnotes
    processFootnotes: function(html) {
        const footnotes = [];
        let footnoteIndex = 1;

        // Process footnote references
        html = html.replace(/\[\^([^\]]+)\]/g, (match, id) => {
            footnotes.push({ id, index: footnoteIndex });
            return `<sup class="footnote-ref"><a href="#footnote-${id}" id="footnote-ref-${id}">${footnoteIndex}</a></sup>`;
        });

        // Add footnote section
        if (footnotes.length > 0) {
            const footnoteHtml = footnotes.map(({ id, index }) => {
                return `
                    <div class="footnote" id="footnote-${id}">
                        <a href="#footnote-ref-${id}" class="footnote-backref">^</a>
                        <span class="footnote-text">Footnote for ${id}</span>
                    </div>
                `;
            }).join('');

            html += `<div class="footnotes"><h4>Footnotes</h4>${footnoteHtml}</div>`;
        }

        return html;
    },

    // Cleanup and final processing
    postProcess: function(html) {
        // Remove empty paragraphs
        html = html.replace(/<p>\s*<\/p>/g, '');
        html = html.replace(/<p><br><\/p>/g, '');

        // Clean up nested formatting
        html = html.replace(/<strong><strong>/g, '<strong>');
        html = html.replace(/<\/strong><\/strong>/g, '</strong>');
        html = html.replace(/<em><em>/g, '<em>');
        html = html.replace(/<\/em><\/em>/g, '</em>');

        return html;
    }
};