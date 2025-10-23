import { Link } from 'react-router-dom';
import { docsStructure } from '../docsConfig';

function Home() {
  return (
    <div className="max-w-6xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl mb-6">
          <span className="text-white text-4xl font-bold">AI</span>
        </div>
        <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
          AI Documentation Hub
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          Comprehensive AI documentation covering 25+ major sections with 1500+ topics,
          from foundational concepts to cutting-edge research.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-1">1500+</div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Topics Covered</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-1">25</div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Main Sections</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-1">75+</div>
          <div className="text-sm text-gray-600 dark:text-gray-400">Notebooks</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-1">500+</div>
          <div className="text-sm text-gray-600 dark:text-gray-400">2024-25 Topics</div>
        </div>
      </div>

      {/* Quick Start */}
      <div className="mb-16">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">Quick Start</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <Link
            to="/doc/README.md"
            className="block p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow"
          >
            <div className="text-2xl mb-3">📚</div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              Getting Started
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Learn about the documentation structure and how to navigate
            </p>
          </Link>

          <Link
            to="/doc/NAVIGATION_INDEX.md"
            className="block p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow"
          >
            <div className="text-2xl mb-3">🗺️</div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              Navigation Index
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Complete guide to all resources and learning paths
            </p>
          </Link>

          <Link
            to="/doc/00_Overview.md"
            className="block p-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow"
          >
            <div className="text-2xl mb-3">🎯</div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              Documentation Overview
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Explore all 25 sections and topics
            </p>
          </Link>
        </div>
      </div>

      {/* Main Sections */}
      <div>
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">
          Documentation Sections
        </h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {docsStructure.slice(0, 15).map((section) => (
            <Link
              key={section.id}
              to={`/doc${section.path}`}
              className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-400 transition-colors"
            >
              <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                {section.title}
              </h3>
              {section.children && section.children.length > 0 && (
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {section.children.length} subsections
                </p>
              )}
            </Link>
          ))}
        </div>

        {docsStructure.length > 15 && (
          <div className="mt-6 text-center">
            <p className="text-gray-600 dark:text-gray-400">
              And {docsStructure.length - 15} more sections...
            </p>
          </div>
        )}
      </div>

      {/* Features */}
      <div className="mt-16 grid md:grid-cols-2 gap-8">
        <div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            📖 What's Inside
          </h3>
          <ul className="space-y-2 text-gray-600 dark:text-gray-400">
            <li>✓ Foundational Machine Learning concepts</li>
            <li>✓ Advanced Deep Learning architectures</li>
            <li>✓ Natural Language Processing & LLMs</li>
            <li>✓ Computer Vision & Generative AI</li>
            <li>✓ AI Ethics, Safety, and Governance</li>
            <li>✓ Industry Applications & Case Studies</li>
          </ul>
        </div>
        <div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            🎓 Learning Paths
          </h3>
          <ul className="space-y-2 text-gray-600 dark:text-gray-400">
            <li>🔰 Beginner: Start with fundamentals</li>
            <li>📊 Intermediate: Build practical skills</li>
            <li>🚀 Advanced: Master complex concepts</li>
            <li>🔬 Researcher: Cutting-edge topics</li>
            <li>💼 Industry: Real-world applications</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Home;
