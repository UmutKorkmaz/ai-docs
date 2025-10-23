# AI Documentation UI

A modern, responsive React-based documentation viewer for the AI Documentation System. Built with React, Vite, and Tailwind CSS, this UI provides an intuitive way to browse and read through 25+ sections of comprehensive AI documentation.

## Features

- **Modern React UI**: Built with React 19 and Vite for fast performance
- **Markdown Rendering**: Full support for GitHub Flavored Markdown with syntax highlighting
- **Responsive Design**: Mobile-friendly interface that works on all devices
- **Dark/Light Mode**: Theme toggle for comfortable reading in any environment
- **Smart Navigation**: Collapsible sidebar with hierarchical documentation structure
- **Code Highlighting**: Beautiful syntax highlighting for code blocks using Highlight.js
- **Fast Search**: Quick navigation through all documentation sections
- **Clean Typography**: Optimized for readability with proper spacing and hierarchy

## Tech Stack

- **React 19.1** - UI framework
- **Vite 7** - Build tool and dev server
- **React Router 7** - Client-side routing
- **Tailwind CSS 3** - Utility-first CSS framework
- **React Markdown** - Markdown rendering
- **Highlight.js** - Code syntax highlighting
- **Rehype/Remark** - Markdown processing plugins

## Project Structure

```
docs-ui/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # Top navigation bar with theme toggle
│   │   ├── Sidebar.jsx         # Navigation sidebar
│   │   ├── Home.jsx            # Landing page
│   │   ├── Layout.jsx          # Main layout wrapper
│   │   └── MarkdownViewer.jsx  # Markdown content renderer
│   ├── App.jsx                 # Main app component with routing
│   ├── docsConfig.js          # Documentation structure configuration
│   ├── index.css              # Global styles and Tailwind directives
│   └── main.jsx               # App entry point
├── public/                     # Markdown files and static assets (auto-copied)
├── copy-docs.js               # Script to copy docs from parent directory
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## Getting Started

### Prerequisites

- Node.js 20 or higher
- npm or yarn

### Installation

1. Navigate to the docs-ui directory:
   ```bash
   cd docs-ui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Copy documentation files:
   ```bash
   npm run copy-docs
   ```

### Development

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Building

Build the production version:
```bash
npm run build
```

The built files will be in the `dist/` directory.

### Preview Production Build

Preview the production build locally:
```bash
npm run preview
```

## Deployment to GitHub Pages

### Automatic Deployment (Recommended)

The repository includes a GitHub Actions workflow that automatically builds and deploys the documentation UI to GitHub Pages when you push to the main branch.

1. Enable GitHub Pages in your repository settings:
   - Go to Settings → Pages
   - Set Source to "GitHub Actions"

2. Push your changes to the main branch:
   ```bash
   git push origin main
   ```

3. The GitHub Actions workflow will automatically:
   - Copy the documentation files
   - Build the React app
   - Deploy to GitHub Pages

Your documentation will be available at: `https://yourusername.github.io/ai-docs/`

### Manual Deployment

You can also deploy manually using the gh-pages package:

```bash
npm run deploy
```

This will:
1. Copy all documentation files
2. Build the production bundle
3. Deploy to the `gh-pages` branch

## Configuration

### Base URL

The app is configured for GitHub Pages deployment at `/ai-docs/`. To change this:

1. Update `vite.config.js`:
   ```javascript
   export default defineConfig({
     base: '/your-repo-name/',
     // ...
   })
   ```

2. Update `App.jsx`:
   ```javascript
   <Router basename="/your-repo-name">
   ```

### Documentation Structure

Edit `src/docsConfig.js` to customize the documentation structure:

```javascript
export const docsStructure = [
  {
    id: 'section-id',
    title: 'Section Title',
    path: '/path/to/doc.md',
    children: [
      // Subsections...
    ]
  },
  // More sections...
];
```

### Styling

The app uses Tailwind CSS. Customize the theme in `tailwind.config.js`:

```javascript
module.exports = {
  theme: {
    extend: {
      // Your customizations
    },
  },
}
```

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run copy-docs` - Copy markdown files from parent directory
- `npm run deploy` - Deploy to GitHub Pages (runs copy-docs and build automatically)

## Adding New Documentation

1. Add your markdown files to the parent directory
2. Run `npm run copy-docs` to copy them to `public/`
3. Update `src/docsConfig.js` to include the new documentation in the navigation

The copy script automatically copies:
- All `.md` files
- Files in `assets/` and `visuals/` directories
- Jupyter notebooks (`.ipynb` files)
- Python files (`.py`)
- Other relevant assets

## Features in Detail

### Markdown Support

The app supports:
- GitHub Flavored Markdown (GFM)
- Tables
- Task lists
- Strikethrough
- Autolinks
- Syntax highlighted code blocks
- Raw HTML in markdown

### Theme Toggle

Click the sun/moon icon in the header to switch between light and dark modes. The theme preference is automatically saved.

### Responsive Navigation

- **Desktop**: Sidebar is always visible
- **Mobile**: Hamburger menu to toggle sidebar
- **Touch**: Swipe gestures supported

### Code Highlighting

Code blocks are automatically highlighted with language detection. Supports 100+ languages including:
- Python
- JavaScript/TypeScript
- Java
- C/C++
- Go
- Rust
- And many more...

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Performance

The app is optimized for performance:
- Fast initial load with Vite
- Code splitting with React Router
- Optimized markdown rendering
- Lazy loading of images
- Minimal bundle size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Troubleshooting

### Documentation files not showing

Run `npm run copy-docs` to ensure all markdown files are copied to the `public/` directory.

### Build errors

1. Clear the cache:
   ```bash
   rm -rf node_modules dist
   npm install
   ```

2. Ensure all dependencies are installed:
   ```bash
   npm install
   ```

### GitHub Pages 404 errors

1. Check that the `base` path in `vite.config.js` matches your repository name
2. Ensure GitHub Pages is enabled in repository settings
3. Verify the GitHub Actions workflow completed successfully

## License

This project is part of the AI Documentation System.

## Links

- [Main Documentation Repository](../)
- [GitHub Pages Deployment](https://yourusername.github.io/ai-docs/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)

---

Built with ❤️ using React, Vite, and Tailwind CSS
