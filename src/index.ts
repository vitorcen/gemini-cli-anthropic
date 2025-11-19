import express from 'express';
import { registerClaudeEndpoints } from './claudeProxy';
import { logger } from './utils/logger';
import { loadConfig, loadEnvironment } from '@a2a/config/config.js';
import { loadSettings } from '@a2a/config/settings.js';
import { SimpleExtensionLoader } from '@google/gemini-cli-core';

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
