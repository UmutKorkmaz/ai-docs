#!/bin/bash
# AI Documentation Cross-Reference System Runner
# This script runs the complete cross-reference generation pipeline

echo "🚀 Starting AI Documentation Cross-Reference System"
echo "=================================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

# Check if required Python packages are installed
echo "📦 Checking dependencies..."
python3 -c "import networkx" 2>/dev/null || {
    echo "⚠️  Installing networkx..."
    pip3 install networkx
}

python3 -c "import numpy" 2>/dev/null || {
    echo "⚠️  Installing numpy..."
    pip3 install numpy
}

# Create output directory if it doesn't exist
mkdir -p cross_reference_output

# Step 1: Generate cross-references
echo ""
echo "🔍 Step 1: Generating cross-references..."
python3 scripts/cross_reference_generator.py

if [ $? -eq 0 ]; then
    echo "✅ Cross-reference generation complete!"
else
    echo "❌ Error in cross-reference generation"
    exit 1
fi

# Step 2: Integrate into documentation
echo ""
echo "📝 Step 2: Integrating cross-references into documentation..."
python3 scripts/integrate_cross_references.py

if [ $? -eq 0 ]; then
    echo "✅ Integration complete!"
else
    echo "❌ Error in integration"
    exit 1
fi

# Summary
echo ""
echo "🎉 Cross-Reference System Setup Complete!"
echo "========================================="
echo ""
echo "📊 Generated Files:"
echo "   - cross_reference_output/ (All generated data)"
echo "   - CROSS_REFERENCE_INDEX.md (Master index)"
echo "   - Updated markdown files with cross-references"
echo ""
echo "🌐 To explore the interactive knowledge navigator:"
echo "   1. Open: components/knowledge_navigator.html"
echo "   2. Or run: python3 -m http.server 8000"
echo "   3. Visit: http://localhost:8000/components/knowledge_navigator.html"
echo ""
echo "📖 For more information, see:"
echo "   - CROSS_REFERENCE_README.md (User guide)"
echo "   - cross_reference_system.md (System documentation)"
echo ""
echo "💡 To update cross-references when documentation changes:"
echo "   Run this script again or use individual scripts:"
echo "   - python3 scripts/cross_reference_generator.py"
echo "   - python3 scripts/integrate_cross_references.py"
echo ""

# Ask if user wants to open the navigator
read -p "Would you like to open the knowledge navigator now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v open &> /dev/null; then
        open components/knowledge_navigator.html
    elif command -v xdg-open &> /dev/null; then
        xdg-open components/knowledge_navigator.html
    else
        echo "Please open components/knowledge_navigator.html in your browser"
    fi
fi

echo "👋 Happy exploring!"