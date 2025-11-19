// src/utils/logger.ts
// A simple logger to replace the one from the original codebase.
export const logger = {
  info: (...args: any[]) => console.log('[INFO]', ...args),
  debug: (...args: any[]) => {
    if (process.env.DEBUG) {
      console.log('[DEBUG]', ...args);
    }
  },
  error: (...args: any[]) => console.error('[ERROR]', ...args),
  warn: (...args: any[]) => console.warn('[WARN]', ...args),
};
