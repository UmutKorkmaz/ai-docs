import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import 'highlight.js/styles/github-dark.css';

function MarkdownViewer({ filePath }) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMarkdown = async () => {
      setLoading(true);
      setError(null);

      try {
        // Remove leading slash if present
        let cleanPath = filePath.startsWith('/') ? filePath.substring(1) : filePath;

        // Ensure .md extension
        if (!cleanPath.endsWith('.md')) {
          // Try adding .md first
          try {
            const response = await fetch(`/${cleanPath}.md`);
            if (response.ok) {
              cleanPath = `${cleanPath}.md`;
            } else {
              // Try as directory with 00_Overview.md
              const overviewResponse = await fetch(`/${cleanPath}/00_Overview.md`);
              if (overviewResponse.ok) {
                cleanPath = `${cleanPath}/00_Overview.md`;
              }
            }
          } catch (e) {
            // Continue with original path
          }
        }

        const response = await fetch(`/${cleanPath}`);

        if (!response.ok) {
          throw new Error(`Failed to load document: ${response.status} ${response.statusText}`);
        }

        const text = await response.text();
        setContent(text);
      } catch (err) {
        console.error('Error loading markdown:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (filePath) {
      fetchMarkdown();
    }
  }, [filePath]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-2">
          Error Loading Document
        </h3>
        <p className="text-red-600 dark:text-red-400">{error}</p>
        <p className="text-sm text-red-500 dark:text-red-300 mt-2">
          Path: {filePath}
        </p>
      </div>
    );
  }

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight, rehypeRaw]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export default MarkdownViewer;
