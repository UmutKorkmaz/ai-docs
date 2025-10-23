import { copyFileSync, mkdirSync, readdirSync, statSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const sourceDir = join(__dirname, '..');
const targetDir = join(__dirname, 'public');

// Patterns to copy
const patterns = [
  '**/*.md',
  'assets/**/*',
  'visuals/**/*',
  'interactive/**/*.ipynb',
  'code_examples/**/*.py',
];

// Directories and files to skip
const skipDirs = new Set([
  'node_modules',
  '.git',
  'docs-ui',
  'documentation-reader',
  'logs',
  'collaboration_system',
  'learning_system',
  'sitemaps',
  'templates',
  '__pycache__',
  '.vscode',
]);

const skipFiles = new Set([
  '.gitignore',
  'package.json',
  'package-lock.json',
  'Dockerfile',
  'docker-compose.yml',
  'requirements.txt',
]);

function shouldSkip(name) {
  return skipDirs.has(name) || name.startsWith('.') || name.endsWith('.pyc');
}

function copyRecursive(src, dest) {
  const stats = statSync(src);

  if (stats.isDirectory()) {
    if (!existsSync(dest)) {
      mkdirSync(dest, { recursive: true });
    }

    const entries = readdirSync(src);

    for (const entry of entries) {
      if (shouldSkip(entry)) continue;

      const srcPath = join(src, entry);
      const destPath = join(dest, entry);

      copyRecursive(srcPath, destPath);
    }
  } else if (stats.isFile()) {
    const ext = src.split('.').pop().toLowerCase();
    const fileName = src.split('/').pop();

    // Copy markdown files and assets
    if (ext === 'md' || src.includes('/assets/') || src.includes('/visuals/') ||
        ext === 'ipynb' || ext === 'py' || ext === 'json' ||
        ext === 'png' || ext === 'jpg' || ext === 'jpeg' || ext === 'gif' || ext === 'svg') {

      if (skipFiles.has(fileName)) return;

      const destDir = dirname(dest);
      if (!existsSync(destDir)) {
        mkdirSync(destDir, { recursive: true });
      }

      try {
        copyFileSync(src, dest);
        console.log(`Copied: ${src.replace(sourceDir, '')}`);
      } catch (err) {
        console.error(`Error copying ${src}:`, err.message);
      }
    }
  }
}

console.log('Copying documentation files...');
console.log(`Source: ${sourceDir}`);
console.log(`Target: ${targetDir}\n`);

// Clean and create target directory
if (existsSync(targetDir)) {
  console.log('Target directory exists, will merge files...\n');
}

copyRecursive(sourceDir, targetDir);

console.log('\nDocumentation files copied successfully!');
