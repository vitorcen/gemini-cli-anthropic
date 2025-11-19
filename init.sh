#!/bin/bash
set -e

echo "=== 1. Initializing git submodules ==="
git submodule update --init --recursive

echo "=== 2. Installing project dependencies ==="
npm install

echo "=== 3. Installing gemini-cli submodule dependencies ==="
cd gemini-cli
npm install
cd ..

echo "=== 4. Starting the application ==="
npm start
