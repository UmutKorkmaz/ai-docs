import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Function to read markdown file and extract title
function extractTitle(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    // Try to find first # heading
    const match = content.match(/^#\s+(.+)$/m);
    if (match) {
      return match[1].trim();
    }
    // Fallback to filename
    return path.basename(filePath, '.md').replace(/_/g, ' ');
  } catch (error) {
    return path.basename(filePath, '.md').replace(/_/g, ' ');
  }
}

// Function to check if a file should be excluded
function shouldExclude(filePath) {
  const excludePatterns = [
    /node_modules/,
    /\.git/,
    /dist/,
    /build/,
    /docs-ui\/public/, // Exclude the copied files
    /interactive/,
    /collaboration_system/,
    /learning_system/,
    /example_projects/,
    /sitemaps/,
    /templates/,
    /scripts\/.*\.md$/,
  ];

  return excludePatterns.some(pattern => pattern.test(filePath));
}

// Function to build directory tree
function buildTree(dirPath, basePath = '') {
  const items = [];

  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });

    // Separate files and directories
    const files = entries.filter(e => e.isFile() && e.name.endsWith('.md'));
    const dirs = entries.filter(e => e.isDirectory() && !e.name.startsWith('.'));

    // Sort entries
    files.sort((a, b) => a.name.localeCompare(b.name));
    dirs.sort((a, b) => a.name.localeCompare(b.name));

    // Process directories first
    for (const dir of dirs) {
      const fullPath = path.join(dirPath, dir.name);
      const relativePath = path.join(basePath, dir.name);

      if (shouldExclude(fullPath)) continue;

      // Check if directory has an overview file
      const overviewFile = path.join(fullPath, '00_Overview.md');
      const readmeFile = path.join(fullPath, 'README.md');
      const hasOverview = fs.existsSync(overviewFile);
      const hasReadme = fs.existsSync(readmeFile);

      const children = buildTree(fullPath, relativePath);

      // Create directory entry
      const dirEntry = {
        id: relativePath.replace(/\//g, '-').replace(/_/g, '-').toLowerCase(),
        title: dir.name.replace(/_/g, ' '),
        path: hasOverview
          ? `/${relativePath}/00_Overview.md`
          : hasReadme
            ? `/${relativePath}/README.md`
            : `/${relativePath}`,
        isDirectory: true,
        children: children
      };

      items.push(dirEntry);
    }

    // Process markdown files
    for (const file of files) {
      const fullPath = path.join(dirPath, file.name);
      const relativePath = path.join(basePath, file.name);

      if (shouldExclude(fullPath)) continue;

      // Skip overview files as they're already included in directory entries
      if (file.name === '00_Overview.md' || file.name === 'README.md') {
        // Only add if there are no subdirectories
        if (items.length === 0 || !items.some(i => i.isDirectory)) {
          const title = extractTitle(fullPath);
          items.push({
            id: relativePath.replace(/\//g, '-').replace(/\./g, '-').replace(/_/g, '-').toLowerCase(),
            title: title,
            path: `/${relativePath}`,
            isDirectory: false
          });
        }
        continue;
      }

      const title = extractTitle(fullPath);

      items.push({
        id: relativePath.replace(/\//g, '-').replace(/\./g, '-').replace(/_/g, '-').toLowerCase(),
        title: title,
        path: `/${relativePath}`,
        isDirectory: false
      });
    }
  } catch (error) {
    console.error(`Error reading directory ${dirPath}:`, error.message);
  }

  return items;
}

// Main function to generate configuration
function generateDocsConfig() {
  const rootDir = path.join(__dirname, '..');
  console.log('Scanning documentation files from:', rootDir);

  // Build the tree
  const docsTree = buildTree(rootDir);

  // Organize into main sections and additional docs
  const mainSections = docsTree.filter(item => {
    const name = item.title.toLowerCase();
    // Main numbered sections (00-25)
    return /^\d{2}[_\s]/.test(item.title) ||
           /^foundational/i.test(name) ||
           /^advanced/i.test(name) ||
           /^natural/i.test(name) ||
           /^computer/i.test(name) ||
           /^generative/i.test(name) ||
           /^ai\s+agents/i.test(name) ||
           /^ethics/i.test(name) ||
           /^mlops/i.test(name) ||
           /^prompt/i.test(name) ||
           /^emerging/i.test(name) ||
           /^specialized/i.test(name) ||
           /^state\s+space/i.test(name) ||
           /^multimodal/i.test(name) ||
           /^social\s+good/i.test(name) ||
           /^policy/i.test(name) ||
           /^human.*ai/i.test(name) ||
           /^entertainment/i.test(name) ||
           /^agriculture/i.test(name) ||
           /^smart\s+cities/i.test(name) ||
           /^aerospace/i.test(name) ||
           /^energy/i.test(name) ||
           /^future/i.test(name);
  });

  const additionalDocs = docsTree.filter(item => {
    const name = item.title.toLowerCase();
    return !mainSections.includes(item) && !item.isDirectory;
  });

  const examplesSections = docsTree.filter(item => {
    const name = item.title.toLowerCase();
    return /industry.*examples/i.test(name) ||
           /companies.*research/i.test(name) ||
           /terminology/i.test(name) ||
           /future.*tech/i.test(name) ||
           /research.*reports/i.test(name) ||
           /deep.*learning.*specialized/i.test(name) ||
           /main.*guides/i.test(name);
  });

  // Generate the config file content
  const configContent = `// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
// Generated on: ${new Date().toISOString()}
// This file is automatically generated by generate-docs-config.js

export const docsStructure = ${JSON.stringify(mainSections, null, 2)};

export const additionalDocs = ${JSON.stringify(additionalDocs, null, 2)};

export const examplesSections = ${JSON.stringify(examplesSections, null, 2)};

// Helper function to flatten the tree for search
export function flattenDocs(docs) {
  const result = [];

  function flatten(items, parentPath = '') {
    for (const item of items) {
      result.push({
        id: item.id,
        title: item.title,
        path: item.path,
        parentPath: parentPath
      });

      if (item.children && item.children.length > 0) {
        flatten(item.children, item.title);
      }
    }
  }

  flatten(docs);
  return result;
}

// Get all documents for search
export function getAllDocs() {
  return [
    ...flattenDocs(docsStructure),
    ...flattenDocs(examplesSections),
    ...additionalDocs.map(d => ({ ...d, parentPath: 'Root' }))
  ];
}
`;

  // Write the configuration file
  const outputPath = path.join(__dirname, 'src', 'docsConfig.js');
  fs.writeFileSync(outputPath, configContent, 'utf-8');

  console.log('\n✅ Documentation configuration generated successfully!');
  console.log(`📄 Output: ${outputPath}`);
  console.log(`📊 Stats:`);
  console.log(`   - Main sections: ${mainSections.length}`);
  console.log(`   - Additional docs: ${additionalDocs.length}`);
  console.log(`   - Examples sections: ${examplesSections.length}`);

  return { mainSections, additionalDocs, examplesSections };
}

// Run the generator
generateDocsConfig();
