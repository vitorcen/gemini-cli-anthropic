// src/utils/logger.ts
// A simple logger to replace the one from the original codebase.
const getTimestamp = () => new Date().toISOString();

export const logger = {
  info: (...args: any[]) => console.log(`[${getTimestamp()}] [INFO]`, ...args),
  debug: (...args: any[]) => {
    if (process.env.DEBUG) {
      console.log(`[${getTimestamp()}] [DEBUG]`, ...args);
    }
  },
  error: (...args: any[]) => console.error(`[${getTimestamp()}] [ERROR]`, ...args),
  warn: (...args: any[]) => console.warn(`[${getTimestamp()}] [WARN]`, ...args),
};
