/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { logger } from './logger.js';
import fs from 'node:fs/promises';
import path from 'node:path';
import type { Content } from '@google/generative-ai';
import { DEFAULT_GEMINI_FLASH_MODEL, DEFAULT_GEMINI_MODEL } from '../../gemini-cli/packages/core/src/index.js';

export const SYNTHETIC_THOUGHT_SIGNATURE = 'skip_thought_signature_validator';
export const MAX_RETRIES = 5;
export const BASE_RETRY_DELAY_MS = 1000;
export const MIN_THOUGHT_LOG_CHARS = 40;

export async function executeWithRetry<T>(
  operation: () => Promise<T>,
  requestId: string,
  model: string
): Promise<T> {
  let lastError: any;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await operation();
    } catch (e: any) {
      lastError = e;
      const message = e.message || '';
      const isQuotaError = message.includes('exhausted your capacity') ||
                           message.includes('quota') ||
                           message.includes('429') ||
                           (message.includes('400') && message.includes('capacity'));

      if (process.env['DEBUG_LOG']) {
        try {
          const logDir = '/tmp/gemini-cli-anthropic';
          try { await fs.mkdir(logDir, { recursive: true }); } catch {}

          const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
          const errorFile = path.join(logDir, `error-${model}-${timestamp}.json`);

          await fs.writeFile(errorFile, JSON.stringify({
            timestamp: new Date().toISOString(),
            requestId,
            model,
            attempt,
            error: {
              name: e.name,
              message: e.message,
              stack: e.stack,
              details: e
            }
          }, null, 2));

          logger.error(`Error generating raw content via API with model ${model}. Full report available at: ${errorFile}`);
        } catch (logError) {
          // Ignore logging errors
        }
      }

      if (!isQuotaError) {
        throw e;
      }

      if (attempt === MAX_RETRIES) {
        logger.error(`[${requestId}] Max retries (${MAX_RETRIES}) exceeded for quota error.`);
        throw e;
      }

      // Parse wait time from "reset after Xs"
      let waitMs = BASE_RETRY_DELAY_MS * Math.pow(2, attempt - 1); // Default exponential backoff
      const match = message.match(/reset after (\d+)s/);
      if (match && match[1]) {
        waitMs = (parseInt(match[1], 10) + 1) * 1000; // Add 1s buffer
      }

      logger.warn(`[${requestId}] Quota exhausted for ${model}. Waiting ${waitMs}ms before retry ${attempt}/${MAX_RETRIES}...`);
      await new Promise(resolve => setTimeout(resolve, waitMs));
    }
  }
  throw lastError;
}

export const formatTokenCount = (value?: number): string =>
  typeof value === 'number' ? value.toLocaleString('en-US') : '0';

export const sumTokenCounts = (...values: Array<number | undefined>): number => {
  let total = 0;
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      total += value;
    }
  }
  return total;
};

export function safeStringify(value: unknown): string {
  if (value === undefined || value === null) {
    return '';
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function serializeArgsForSse(args: unknown): string {
  const serialized = safeStringify(args);
  if (!serialized || serialized === 'undefined') {
    return '{}';
  }
  try {
    JSON.parse(serialized);
    return serialized;
  } catch {
    return '{}';
  }
}

/**
 * Clean JSON Schema by removing metadata fields that Gemini API doesn't accept
 */
export function cleanSchema(schema: any): any {
  if (!schema || typeof schema !== 'object') return schema;

  // Remove JSON Schema metadata fields
  const rest: Record<string, any> = { ...schema };
  delete rest.$schema;
  delete rest.$id;
  delete rest.$ref;
  delete rest.$comment;
  delete rest.definitions;
  delete rest.$defs;

  // Gemini API rejects exclusiveMinimum/exclusiveMaximum; map to min/max when possible.
  if (typeof rest.exclusiveMinimum === 'number') {
    const exclusiveMin = rest.exclusiveMinimum;
    delete rest.exclusiveMinimum;
    if (typeof rest.minimum !== 'number') {
      rest.minimum = rest.type === 'integer' ? exclusiveMin + 1 : exclusiveMin;
    } else if (rest.type === 'integer') {
      rest.minimum = Math.max(rest.minimum, exclusiveMin + 1);
    } else {
      rest.minimum = Math.max(rest.minimum, exclusiveMin);
    }
  }
  if (typeof rest.exclusiveMaximum === 'number') {
    const exclusiveMax = rest.exclusiveMaximum;
    delete rest.exclusiveMaximum;
    if (typeof rest.maximum !== 'number') {
      rest.maximum = rest.type === 'integer' ? exclusiveMax - 1 : exclusiveMax;
    } else if (rest.type === 'integer') {
      rest.maximum = Math.min(rest.maximum, exclusiveMax - 1);
    } else {
      rest.maximum = Math.min(rest.maximum, exclusiveMax);
    }
  }

  // Recursively clean nested objects
  const cleaned: any = {};
  for (const [key, value] of Object.entries(rest)) {
    if (key === 'properties' && typeof value === 'object') {
      cleaned[key] = {};
      for (const [propKey, propValue] of Object.entries(value as any)) {
        cleaned[key][propKey] = cleanSchema(propValue);
      }
    } else if (key === 'items' && typeof value === 'object') {
      cleaned[key] = cleanSchema(value);
    } else if (key === 'additionalProperties' && typeof value === 'object') {
      cleaned[key] = cleanSchema(value);
    } else if (Array.isArray(value)) {
      cleaned[key] = value.map(item => typeof item === 'object' ? cleanSchema(item) : item);
    } else {
      cleaned[key] = value;
    }
  }

  return cleaned;
}

/**
 * Map Claude model names to Gemini model names
 */
export function mapModelName(requestedModel: string | undefined): string {
  if (!requestedModel) return DEFAULT_GEMINI_FLASH_MODEL;

  const lowerModel = requestedModel.toLowerCase();

  // Models containing "sonnet" or "opus" -> DEFAULT_GEMINI_MODEL
  if (lowerModel.includes('sonnet') || lowerModel.includes('opus')) {
    return DEFAULT_GEMINI_MODEL;
  }

  // Models containing "haiku" -> DEFAULT_GEMINI_FLASH_MODEL
  // Claude code auto req the Kaiku model based on user input to determine if it's a new topic and extract a title
  if (lowerModel.includes('haiku')) {
    return DEFAULT_GEMINI_FLASH_MODEL;
  }

  // Other claude-* models -> flash
  if (requestedModel.startsWith('claude-')) {
    return DEFAULT_GEMINI_FLASH_MODEL;
  }

  // Pass through everything else (gemini-*, gpt-*, etc.)
  return requestedModel;
}

/**
 * Filter out thought parts and thoughtSignature from response to save context space
 */
export function filterThoughtParts(parts: any[]): any[] {
  return parts
    // Drop explicit thought blocks so they don't leak into the user-visible stream/history
    .filter(p => !(p as any)?.thought)
    .map(p => {
      if (p && typeof p === 'object') {
        // Strip signatures from responses; requests will re-add synthetic ones as needed
        const { thoughtSignature, thought_signature, ...rest } = p as any;
        return rest;
      }
      return p;
    });
}

export function logThoughtParts(parts: any[], requestId: string, phase: 'stream' | 'non-stream' | 'stream-fallback', forceLog = false): void {
  if (!process.env['TEXT_LOG']) return;

  const thoughtBlocks = forceLog ? parts : parts.filter(p => {
    const block = p as any;
    // Use Object.keys for robust check in case of getter/setter issues
    const hasSignature = Object.keys(block).includes('thoughtSignature') || (block as any).thoughtSignature;
    return block?.thought || block?.type === 'thinking' || hasSignature;
  });

  if (thoughtBlocks.length === 0) return;

  const textParts = thoughtBlocks.map(block => {
    const b = block as any;
    // Try to extract text from text or content field
    if (typeof b.text === 'string') return b.text;
    if (typeof b.content === 'string') return b.content;
    // Check if thought field itself is a string
    if (typeof b.thought === 'string') return b.thought;

    // Fallback: stringify the object as JSON, but strip signature to keep log clean
    try {
      const cleanBlock = { ...b };
      delete cleanBlock.thoughtSignature;
      delete cleanBlock.thought_signature;
      return JSON.stringify(cleanBlock);
    } catch {
      return JSON.stringify(b);
    }
  }).filter(Boolean);

  if (textParts.length === 0) {
    if (forceLog) logger.info(`[${requestId}] TEXT_LOG (Empty thought chunk)`);
    return;
  }

  const combinedText = textParts.join(' ');

  if (!combinedText.trim()) {
    // if (forceLog) logger.info(`[${requestId}] TEXT_LOG (Whitespace-only thought chunk)`);
    return;
  }

  let displayedText = combinedText;
  // Collapse whitespace
  displayedText = displayedText.replace(/\s+/g, ' ').trim();

  if (displayedText.length > 200) {
    displayedText = `${displayedText.slice(0, 130)}...${displayedText.slice(-70)}`;
  }

  // Avoid noisy logs for tiny fragments; wait for a meaningful chunk.
  if (displayedText.length < MIN_THOUGHT_LOG_CHARS) {
    return;
  }

  logger.info(`[${requestId}] TEXT_LOG ${displayedText}`);
}

/**
 * Write debug log when DEBUG_LOG is set
 */
export async function writeDebugLog(
  requestId: string,
  type: string,
  data: Record<string, unknown>,
): Promise<void> {
  if (!process.env['DEBUG_LOG']) {
    return;
  }

  try {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    // Use specific directory for logs
    const logDir = '/tmp/gemini-cli-anthropic';

    // Ensure directory exists (quietly)
    try {
      await fs.mkdir(logDir, { recursive: true });
    } catch {
      // Ignore mkdir errors
    }

    const filename = `claude-proxy-${type}-${requestId}-${timestamp}.json`;
    const filepath = path.join(logDir, filename);
    await fs.writeFile(filepath, JSON.stringify(data, null, 2));
    logger.debug(`[${requestId}] Debug log written to: ${filepath}`);
  } catch (error) {
    logger.error(`[${requestId}] Failed to write debug log:`, error);
  }
}

/**
 * Merge consecutive contents from the same role to satisfy Gemini API requirements
 */
export function mergeConsecutiveContents(contents: Content[]): Content[] {
  if (contents.length === 0) return [];

  const merged: Content[] = [{
    role: contents[0].role,
    parts: [...(contents[0].parts || [])]
  }];

  for (let i = 1; i < contents.length; i++) {
    const current = contents[i];
    const last = merged[merged.length - 1];

    if (current.role === last.role) {
      // Same role, merge parts
      last.parts = [...(last.parts || []), ...(current.parts || [])];
    } else {
      // Different role, add new content
      merged.push({
        role: current.role,
        parts: [...(current.parts || [])]
      });
    }
  }

  return merged;
}

export function collectDebugRequestStats(contents: Content[]) {
  let totalParts = 0;
  let toolUseCount = 0;
  let toolResultCount = 0;
  let lastUserTextLen = 0;
  let lastAssistantTextLen = 0;
  let hasSystemReminderInUser = false;

  for (const content of contents) {
    const parts = content.parts || [];
    totalParts += parts.length;

    if (content.role === 'user') {
      for (const p of parts) {
        if (typeof (p as any).text === 'string') {
          lastUserTextLen = (p as any).text.length;
          if ((p as any).text.includes('<system-reminder>')) {
            hasSystemReminderInUser = true;
          }
        }
      }
    }

    if (content.role === 'model') {
      for (const p of parts) {
        if (typeof (p as any).text === 'string') {
          lastAssistantTextLen = (p as any).text.length;
        }
      }
    }

    for (const p of parts) {
      if ((p as any).functionCall) {
        toolUseCount++;
      }
      if ((p as any).functionResponse) {
        toolResultCount++;
      }
    }
  }

  const totalBytes = Buffer.byteLength(JSON.stringify(contents), 'utf8');

  return {
    messageCount: contents.length,
    totalParts,
    totalBytes,
    lastUserTextLen,
    lastAssistantTextLen,
    hasSystemReminderInUser,
    toolUseCount,
    toolResultCount,
  };
}

export function summarizeContentsForDebug(contents: Content[]) {
  const summaries = contents.map((content, idx) => {
    const parts = content.parts || [];
    let totalBytes = 0;
    let textBytes = 0;
    let functionCalls = 0;
    let functionResponses = 0;
    let maxPartBytes = 0;

    for (const p of parts) {
      const serialized = JSON.stringify(p) || '';
      const size = Buffer.byteLength(serialized, 'utf8');
      totalBytes += size;
      maxPartBytes = Math.max(maxPartBytes, size);
      if ((p as any).text !== undefined) {
        textBytes += Buffer.byteLength(String((p as any).text), 'utf8');
      }
      if ((p as any).functionCall) functionCalls++;
      if ((p as any).functionResponse) functionResponses++;
    }

    return {
      index: idx,
      role: content.role,
      parts: parts.length,
      totalBytes,
      textBytes,
      maxPartBytes,
      functionCalls,
      functionResponses,
    };
  });

  const largestParts: Array<{
    contentIndex: number;
    partIndex: number;
    role: string;
    bytes: number;
    preview: string;
  }> = [];

  contents.forEach((content, ci) => {
    (content.parts || []).forEach((p, pi) => {
      const serialized = JSON.stringify(p) || '';
      const size = Buffer.byteLength(serialized, 'utf8');
      const previewSource =
        typeof (p as any).text === 'string'
          ? (p as any).text
          : serialized;
      const preview =
        previewSource.length > 200
          ? `${previewSource.slice(0, 200)}...`
          : previewSource;

      largestParts.push({
        contentIndex: ci,
        partIndex: pi,
        role: content.role,
        bytes: size,
        preview,
      });
    });
  });

  largestParts.sort((a, b) => b.bytes - a.bytes);

  return {
    summaries,
    largestParts: largestParts.slice(0, 10),
  };
}
