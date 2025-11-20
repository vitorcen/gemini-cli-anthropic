#!/usr/bin/env node
import express from 'express';
import { registerClaudeEndpoints } from './claudeProxy.js';
import { logger } from './utils/logger.js';
import { loadConfig, loadEnvironment } from '../gemini-cli/packages/a2a-server/src/config/config.js';
import { loadSettings } from '../gemini-cli/packages/a2a-server/src/config/settings.js';
import { SimpleExtensionLoader } from '../gemini-cli/packages/core/src/index.js';

async function main() {
  // Set default environment variables if not present
  if (!process.env.USE_CCPA && !process.env.GEMINI_API_KEY) {
    process.env.USE_CCPA = 'true';
    logger.info('Defaulting to USE_CCPA=true for authentication.');
  }
  // Load environment variables from .env file
  loadEnvironment();

  const app = express();
  const port = process.env.PORT || 41242;

  app.use(express.json({ limit: '50mb' }));

  const apiRouter = express.Router();

  // Create an extension loader with no extensions
  const extensionLoader = new SimpleExtensionLoader([]);

  // Load the default configuration
  const workspaceDir = process.cwd();

  // Parse command line arguments for --log-dir
  const args = process.argv.slice(2);
  const logDirArg = args.find(arg => arg.startsWith('--log-dir='));
  if (logDirArg) {
    const logDir = logDirArg.split('=')[1];
    process.env.CLAUDE_PROXY_LOG_DIR = logDir;
    try {
      const fs = await import('node:fs/promises');
      await fs.mkdir(logDir, { recursive: true });
      logger.info(`Logging detailed conversation history to: ${logDir}`);
    } catch (e) {
      logger.error(`Failed to create log directory ${logDir}:`, e);
    }
  }

  const settings = loadSettings(workspaceDir);
  const config = await loadConfig(settings, extensionLoader, `claude-proxy-init-${Date.now()}`);

  registerClaudeEndpoints(apiRouter, config);

  app.use('/v1', apiRouter);

  app.listen(port, () => {
    logger.info(`Claude-to-Gemini proxy server listening on port ${port}`);
  });
}

main().catch(err => {
  logger.error('Failed to start server:', err);
});
