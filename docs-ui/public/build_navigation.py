#!/usr/bin/env python3
"""
Build Script for AI Documentation Navigation System
===================================================

This script builds and initializes the complete navigation system for the AI documentation.
It generates all necessary files and demonstrates the system functionality.

Author: AI Documentation Team
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add the components directory to the path
components_path = Path(__file__).parent / "components"
sys.path.insert(0, str(components_path))

try:
    from navigation_integration import AIDocumentationNavigationApp
    from navigation_system import AIDocumentationNavigator
    from navigation_ui import NavigationUIRenderer
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all navigation files are in the components/ directory")
    sys.exit(1)


def main():
    """Main build function."""
    print("🚀 Building AI Documentation Navigation System")
    print("=" * 60)

    # Check if we're in the right directory
    base_path = Path("/Users/dtumkorkmaz/Projects/ai-docs")
    if not base_path.exists():
        print(f"❌ Base path not found: {base_path}")
        sys.exit(1)

    try:
        # Initialize the navigation app
        print("📦 Initializing navigation system...")
        app = AIDocumentationNavigationApp(str(base_path))

        # Build static files
        print("🔨 Building static files...")
        app.build_static_files()

        # Print system statistics
        print("\n📊 System Statistics:")
        stats = app.get_user_statistics()
        print(f"   ✅ Total sections: {stats['total_sections']}")
        print(f"   ✅ Learning paths: {len(app.navigator.learning_paths)}")

        total_notebooks = sum(s.interactive_notebooks for s in app.navigator.sections.values())
        print(f"   ✅ Interactive notebooks: {total_notebooks}")

        # Show enabled features
        print(f"\n🎯 Enabled Features:")
        features = app.config.get("navigation_features", {})
        for feature, config in features.items():
            if isinstance(config, dict) and config.get("enabled"):
                print(f"   ✅ {feature.replace('_', ' ').title()}")
            elif isinstance(config, bool) and config:
                print(f"   ✅ {feature.replace('_', ' ').title()}")

        # Show file structure
        print(f"\n📁 Generated Files:")
        generated_files = [
            "navigation.html - Main navigation page",
            "components/css/navigation.css - Responsive styles",
            "components/js/navigation.js - Interactive functionality",
            "components/templates/search_results.html - Search template",
            "components/navigation_config.json - Configuration",
            "components/user_data.json - User progress data"
        ]

        for file_desc in generated_files:
            print(f"   📄 {file_desc}")

        # Show learning paths
        print(f"\n🎓 Available Learning Paths:")
        for path_id, path in app.navigator.learning_paths.items():
            print(f"   📚 {path.name} ({path.estimated_duration}h, Level {path.difficulty_level})")

        # Show quick start instructions
        print(f"\n🚀 Quick Start:")
        print(f"   1. Open navigation.html in your browser")
        print(f"   2. Use Ctrl+K to search and navigate")
        print(f"   3. Choose a learning path that fits your goals")
        print(f"   4. Track your progress as you learn")

        print(f"\n🔗 Important Links:")
        print(f"   📖 Navigation: navigation.html")
        print(f"   ⚙️  Config: components/navigation_config.json")
        print(f"   👤 User Data: components/user_data.json")
        print(f"   📚 Documentation: components/README.md")

        # Demo some functionality
        print(f"\n🎮 Demo:")
        print(f"   ✅ Navigation system initialized with {len(app.navigator.sections)} sections")
        print(f"   ✅ Cross-references built between related topics")
        print(f"   ✅ Knowledge graph created for conceptual relationships")
        print(f"   ✅ User progress tracking enabled")
        print(f"   ✅ Search functionality configured")
        print(f"   ✅ Learning paths mapped and ready")

        print(f"\n✨ Navigation system build completed successfully!")
        print(f"   🌟 Ready for interactive AI documentation exploration!")

    except Exception as e:
        print(f"❌ Error building navigation system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()