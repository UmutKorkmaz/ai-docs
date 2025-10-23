import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAllDocs } from '../docsConfig';

function SearchBar({ isMobile = false }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    function handleClickOutside(event) {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    const allDocs = getAllDocs();
    const searchTerm = query.toLowerCase();

    const filtered = allDocs
      .filter(doc => {
        const titleMatch = doc.title.toLowerCase().includes(searchTerm);
        const pathMatch = doc.path.toLowerCase().includes(searchTerm);
        const parentMatch = doc.parentPath?.toLowerCase().includes(searchTerm);
        return titleMatch || pathMatch || parentMatch;
      })
      .slice(0, 10);

    setResults(filtered);
    setIsOpen(filtered.length > 0);
    setSelectedIndex(0);
  }, [query]);

  const handleSelect = (path) => {
    navigate(`/doc${path}`);
    setQuery('');
    setIsOpen(false);
  };

  const handleKeyDown = (e) => {
    if (!isOpen || results.length === 0) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % results.length);
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + results.length) % results.length);
        break;
      case 'Enter':
        e.preventDefault();
        if (results[selectedIndex]) {
          handleSelect(results[selectedIndex].path);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        setQuery('');
        break;
      default:
        break;
    }
  };

  return (
    <div ref={searchRef} className={`relative ${isMobile ? 'w-full' : 'w-96'}`}>
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => query.length >= 2 && results.length > 0 && setIsOpen(true)}
          placeholder="Search documentation..."
          className="w-full px-4 py-2 pl-10 pr-4 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent"
        />
        <svg
          className="absolute left-3 top-2.5 w-5 h-5 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>
      </div>

      {isOpen && results.length > 0 && (
        <div className="absolute z-50 w-full mt-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg max-h-96 overflow-y-auto">
          {results.map((doc, index) => (
            <button
              key={doc.id}
              onClick={() => handleSelect(doc.path)}
              onMouseEnter={() => setSelectedIndex(index)}
              className={`w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${
                index === selectedIndex
                  ? 'bg-gray-100 dark:bg-gray-700'
                  : ''
              } ${index > 0 ? 'border-t border-gray-200 dark:border-gray-700' : ''}`}
            >
              <div className="font-medium text-gray-900 dark:text-white">
                {doc.title}
              </div>
              {doc.parentPath && (
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {doc.parentPath}
                </div>
              )}
              <div className="text-xs text-gray-400 dark:text-gray-500 mt-1 font-mono">
                {doc.path}
              </div>
            </button>
          ))}
        </div>
      )}

      {isOpen && results.length === 0 && query.length >= 2 && (
        <div className="absolute z-50 w-full mt-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-4">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No results found for "{query}"
          </p>
        </div>
      )}
    </div>
  );
}

export default SearchBar;
