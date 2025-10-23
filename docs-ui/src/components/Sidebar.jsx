import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { docsStructure, additionalDocs } from '../docsConfig';

function Sidebar({ isOpen, onClose }) {
  const [expandedSections, setExpandedSections] = useState({});
  const location = useLocation();

  const toggleSection = (id) => {
    setExpandedSections(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const isActive = (path) => {
    const docPath = path.replace(/^\//, '');
    return location.pathname === `/doc/${docPath}` || location.pathname === `/doc/${docPath}.md`;
  };

  const renderDocItem = (doc) => {
    const hasChildren = doc.children && doc.children.length > 0;
    const isExpanded = expandedSections[doc.id];
    const active = isActive(doc.path);

    return (
      <div key={doc.id} className="mb-1">
        <div className="flex items-center">
          {hasChildren && (
            <button
              onClick={() => toggleSection(doc.id)}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded mr-1"
            >
              <svg
                className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
            </button>
          )}
          <Link
            to={`/doc${doc.path}`}
            onClick={onClose}
            className={`flex-1 px-3 py-2 rounded-md text-sm transition-colors ${
              active
                ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 font-semibold'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
            } ${!hasChildren ? 'ml-5' : ''}`}
          >
            {doc.title}
          </Link>
        </div>
        {hasChildren && isExpanded && (
          <div className="ml-4 mt-1 border-l-2 border-gray-200 dark:border-gray-700 pl-2">
            {doc.children.map(child => renderDocItem(child))}
          </div>
        )}
      </div>
    );
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 h-full w-72 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 transform transition-transform duration-300 ease-in-out z-50 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } lg:translate-x-0 lg:static overflow-y-auto`}
      >
        <div className="p-4">
          <div className="flex items-center justify-between mb-6">
            <Link to="/" className="flex items-center space-x-2" onClick={onClose}>
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white text-xl font-bold">AI</span>
              </div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">AI Docs</h1>
            </Link>
            <button
              onClick={onClose}
              className="lg:hidden p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Quick Links */}
          <div className="mb-6">
            <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
              Quick Links
            </h3>
            <div className="space-y-1">
              {additionalDocs.map(doc => (
                <Link
                  key={doc.id}
                  to={`/doc${doc.path}`}
                  onClick={onClose}
                  className={`block px-3 py-2 rounded-md text-sm transition-colors ${
                    isActive(doc.path)
                      ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 font-semibold'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                  }`}
                >
                  {doc.title}
                </Link>
              ))}
            </div>
          </div>

          {/* Main Documentation */}
          <div>
            <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
              Documentation
            </h3>
            <div className="space-y-1">
              {docsStructure.map(doc => renderDocItem(doc))}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
