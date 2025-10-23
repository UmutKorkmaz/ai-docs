import { Link } from 'react-router-dom';

function Breadcrumbs({ path }) {
  if (!path) return null;

  // Split path and filter out empty strings
  const pathParts = path
    .split('/')
    .filter(part => part && part !== 'doc')
    .map(part => part.replace('.md', ''));

  // If no parts, return null
  if (pathParts.length === 0) return null;

  // Build breadcrumb items
  const breadcrumbs = [
    { label: 'Home', path: '/' },
  ];

  let currentPath = '';
  pathParts.forEach((part, index) => {
    currentPath += `/${part}`;
    const isLast = index === pathParts.length - 1;

    // Format label: replace underscores and hyphens with spaces, capitalize
    let label = part
      .replace(/_/g, ' ')
      .replace(/-/g, ' ')
      .replace(/^\d+\s*/, ''); // Remove leading numbers

    // Capitalize words
    label = label
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');

    breadcrumbs.push({
      label,
      path: isLast ? null : `/doc${currentPath}`,
    });
  });

  return (
    <nav className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400 mb-6 flex-wrap">
      {breadcrumbs.map((crumb, index) => (
        <div key={index} className="flex items-center">
          {index > 0 && (
            <svg
              className="w-4 h-4 mx-2 text-gray-400"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                clipRule="evenodd"
              />
            </svg>
          )}
          {crumb.path ? (
            <Link
              to={crumb.path}
              className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              {crumb.label}
            </Link>
          ) : (
            <span className="font-medium text-gray-900 dark:text-white">
              {crumb.label}
            </span>
          )}
        </div>
      ))}
    </nav>
  );
}

export default Breadcrumbs;
