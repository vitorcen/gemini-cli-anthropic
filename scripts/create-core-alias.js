/**
 * @license
 * Copyright 2025
 */
import fs from 'node:fs';
import path from 'node:path';

// Create a tiny re-export so the bundled CLI can resolve @google/gemini-cli-core
const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const distRoot = path.join(repoRoot, 'dist');
const targetDir = path.join(distRoot, 'node_modules', '@google', 'gemini-cli-core');
const sourceEntry = path.relative(targetDir, path.join(distRoot, 'gemini-cli', 'packages', 'core', 'src', 'index.js'));

fs.mkdirSync(targetDir, { recursive: true });

fs.writeFileSync(
  path.join(targetDir, 'package.json'),
  JSON.stringify(
    {
      name: '@google/gemini-cli-core',
      type: 'module',
      main: 'index.js',
    },
    null,
    2,
  ),
);

fs.writeFileSync(
  path.join(targetDir, 'index.js'),
  `export * from '${sourceEntry.startsWith('.') ? sourceEntry : `./${sourceEntry}`}';\n`,
);

console.log(`Created alias for @google/gemini-cli-core at ${targetDir}`);
