/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type express from 'express';
import { v4 as uuidv4 } from 'uuid';
import type { Config } from '../gemini-cli/packages/core/src/index.js';
import { DEFAULT_GEMINI_FLASH_MODEL, SimpleExtensionLoader, DEFAULT_GEMINI_MODEL } from '../gemini-cli/packages/core/src/index.js';
import { logger } from './utils/logger.js';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const SYNTHETIC_THOUGHT_SIGNATURE = 'skip_thought_signature_validator';

const MAX_RETRIES = 5;
const BASE_RETRY_DELAY_MS = 1000;

async function executeWithRetry<T>(
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

      if (process.env['DEBUG_LOG_REQUESTS']) {
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

import type { Content } from '@google/generative-ai';
// logger imported at the top
// fs, os, path imported at the top

const formatTokenCount = (value?: number): string =>
  typeof value === 'number' ? value.toLocaleString('en-US') : '0';

const sumTokenCounts = (...values: Array<number | undefined>): number => {
  let total = 0;
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      total += value;
    }
  }
  return total;
};

interface ClaudeContentBlock {
  type: 'text' | 'tool_use' | 'tool_result';
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, any>;
  tool_use_id?: string;
  tool_id?: string;
  content?: string | Record<string, any>;
  thoughtSignature?: string;
}

function safeStringify(value: unknown): string {
  if (value === undefined || value === null) {
    return '';
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function serializeArgsForSse(args: unknown): string {
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

interface ClaudeMessage {
  role: 'user' | 'assistant';
  content: ClaudeContentBlock[] | string;
}

interface ClaudeRequest {
  model?: string;
  messages: ClaudeMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
  system?: Array <{
    type: 'text';
    text: string;
  }> | string;
  tools?: Array <{
    name: string;
    description?: string;
    input_schema?: {
      type: string;
      properties?: Record<string, any>;
      required?: string[];
    };
  }>;
}

/**
 * Clean JSON Schema by removing metadata fields that Gemini API doesn't accept
 */
function cleanSchema(schema: any): any {
  if (!schema || typeof schema !== 'object') return schema;

  // Remove JSON Schema metadata fields
  const { $schema, $id, $ref, $comment, definitions, $defs, ...rest } = schema;

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
function mapModelName(requestedModel: string | undefined): string {
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

function filterThoughtParts(parts: any[]): any[] {
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

function logThoughtParts(parts: any[], requestId: string, phase: 'stream' | 'non-stream', forceLog = false): void {
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
    if (forceLog) logger.info(`[${requestId}] TEXT_LOG (Whitespace-only thought chunk)`);
    return;
  }

  let displayedText = combinedText;
  // Collapse whitespace
  displayedText = displayedText.replace(/\s+/g, ' ').trim();

  if (displayedText.length > 200) {
    displayedText = `${displayedText.slice(0, 130)}...${displayedText.slice(-70)}`;
  }

  logger.info(`[${requestId}] TEXT_LOG ${displayedText}`);
}

/**
 * Write debug log when DEBUG_LOG is set
 */
async function writeDebugLog(
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
function mergeConsecutiveContents(contents: Content[]): Content[] {
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

function collectDebugRequestStats(contents: Content[]) {
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

function summarizeContentsForDebug(contents: Content[]) {
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

export function registerClaudeEndpoints(app: express.Router, defaultConfig: Config) {
  // Claude-compatible /v1/messages endpoint
  app.post('/messages', async (req: express.Request, res: express.Response) => {
    const headerRequestId = req.headers['x-request-id'];
    const requestId =
        typeof headerRequestId === 'string' && headerRequestId.trim().length > 0
          ? headerRequestId
          : uuidv4();
    try {
      const body = (req.body ?? {}) as ClaudeRequest;

      // Debug: Log tools if present
      if (body.tools !== undefined) {
        logger.debug(`[${requestId}] Received tools: ${JSON.stringify(body.tools)}`);
      }

      if (!Array.isArray(body.messages)) {
        throw new Error('`messages` must be an array.');
      }
      const stream = Boolean(body.stream);
      const model = mapModelName(body.model);

      // Check for X-Working-Directory header to support per-request working directory
      const workingDirectory = req.headers['x-working-directory'] as string | undefined;
      let config = defaultConfig;

      if (workingDirectory) {
        // Create a new config with the specified working directory
        const { loadConfig } = await import('../gemini-cli/packages/a2a-server/src/config/config.js');
        const { loadSettings } = await import('../gemini-cli/packages/a2a-server/src/config/settings.js');
        const settings = loadSettings(workingDirectory);
        config = await loadConfig(settings, new SimpleExtensionLoader([]), req.headers['x-request-id'] as string || Date.now().toString());
      }

      // Extract parameters
      const temperature = body.temperature;
      const topP = body.top_p;
      const maxOutputTokens = body.max_tokens || 4096;

      // Build Gemini-compatible contents array with tool support
      const unprocessedContents: Content[] = [];
      const toolUseMap = new Map<string, string>(); // tool_use_id -> name

      for (const message of body.messages) {
        if (typeof message.content === 'string') {
          // Simple text message
          unprocessedContents.push({
            role: message.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: message.content }]
          });
        } else {
          // Structured content blocks
          const parts: any[] = [];

          for (const block of message.content) {
            // Drop any explicit thinking/thought blocks so they don't re-enter history
            if ((block as any).thought !== undefined || (block as any).type === 'thinking') {
              continue;
            }

            if (block.type === 'text' && block.text) {
              // Skip injected system-reminder payloads that may get duplicated into user turns
              // if (message.role === 'user' && block.text.includes('<system-reminder>')) {
              //   continue;
              // }
              parts.push({ text: block.text });
            } else if (block.type === 'tool_use' && (block.id || (block as any).tool_id || true) && block.name) {
              // Assistant tool call -> Gemini functionCall
              const toolId = block.id || (block as any).tool_id || `toolu_${uuidv4()}`;
              const args = block.input ?? {};
              if (toolId) {
                toolUseMap.set(toolId, block.name);
              }
              const explicitSignature =
                (block as any).thoughtSignature ||
                (block as any).thought_signature;
              const part: any = {
                functionCall: {
                  name: block.name,
                  args,
                },
              };
              // Preserve explicit signature, otherwise provide synthetic to satisfy API
              const signature = explicitSignature || SYNTHETIC_THOUGHT_SIGNATURE;
              part.thoughtSignature = signature;
              (part as any).thought_signature = signature;
              parts.push(part);
            } else if (block.type === 'tool_result') {
              // User tool result -> Gemini functionResponse
              const toolId = block.tool_use_id || (block as any).tool_id;
              const toolName = toolId ? (toolUseMap.get(toolId) ?? block.name) : block.name;
              const resolvedName = toolName || 'unknown';
              const resultPayload =
                typeof block.content === 'string' ? { result: block.content } : { result: block.content };
              parts.push({
                functionResponse: {
                  name: resolvedName,
                  response: resultPayload,
                },
              });
            }
          }

          if (parts.length > 0) {
            unprocessedContents.push({
              role: message.role === 'assistant' ? 'model' : 'user',
              parts
            });
          }
        }
      }

      // Different processing logic for streaming vs non-streaming requests
      // Non-streaming requests (e.g., title generation) should not merge or modify messages
      // Streaming requests (normal conversation) should merge consecutive same-role messages
      let contents: Content[];
      if (stream) {
        // Streaming: merge consecutive contents from the same role to ensure API compliance
        contents = mergeConsecutiveContents(unprocessedContents);
      } else {
        // Non-streaming: use messages as-is without merging to avoid contamination
        contents = unprocessedContents;
      }

      // Ensure we have at least one message to send
      if (contents.length === 0) {
        throw new Error('No valid messages to send to the model');
      }

      // Handle system prompt and tools
      let systemInstruction: Content | undefined;
      if (body.system) {
        const systemContent =
          typeof body.system === 'string'
            ? body.system
            : body.system.map((s: any) => s.text).join('\n');
        systemInstruction = { parts: [{ text: systemContent }], role: 'system' };
      }

      const tools = body.tools && body.tools.length > 0
        ? [{ functionDeclarations: body.tools.map(t => ({
            name: t.name,
            description: t.description || '',
            parameters: cleanSchema(t.input_schema)
          })) }]
        : undefined;

      if (stream) {
        // SSE streaming response
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        // @ts-ignore
        res.flushHeaders && res.flushHeaders();

        let keepAliveInterval: NodeJS.Timeout | undefined;

        try {
          // Use ContentGenerator directly to bypass Turn logic
          const textSizeKB = (Buffer.byteLength(JSON.stringify(contents), 'utf8') / 1024).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
          logger.info(
            `[${requestId}] Sending request model=${model} stream=true text=${textSizeKB}KB`,
          );

          // Log request details if DEBUG_LOG_REQUESTS is set
          if (process.env['DEBUG_LOG_REQUESTS']) {
            try {
              const logDir = '/tmp/gemini-cli-anthropic';
              try { await fs.mkdir(logDir, { recursive: true }); } catch {}

              const logPath = path.join(logDir, `gemini-cli-anthropic-${requestId}.txt`);
              const logContent = `[INFO] ${new Date().toLocaleString()} -- [${requestId}] Sending request model=${model} stream=true text=${textSizeKB}KB\n` +
                `System: ${JSON.stringify(systemInstruction, null, 2)}\n` +
                `Tools: ${JSON.stringify(tools, null, 2)}\n` +
                `Contents:\n${JSON.stringify(contents, null, 2)}\n\n`;

              await fs.appendFile(logPath, logContent);
            } catch (e) {
              // Ignore logging errors
            }
          }

          if (process.env['DEBUG_LOG']) {
            await writeDebugLog(requestId, 'preflight', {
              model,
              stream: true,
              stats: collectDebugRequestStats(contents),
              contentDebug: summarizeContentsForDebug(contents),
              systemInstruction,
              tools,
            });
          }

          const streamGen = await executeWithRetry(
            async () => {
              const client = config.getGeminiClient() as any;
              return client.rawGenerateContentStream(
                contents,
                {
                  temperature,
                  topP,
                  maxOutputTokens,
                  ...(tools && { tools: tools as any }),
                  ...(systemInstruction && { systemInstruction: systemInstruction as any }),
                },
                new AbortController().signal,
                model
              ) as Promise<AsyncGenerator<any>>;
            },
            requestId,
            model
          );

          let outputTokens = 0;
          let inputTokens = 0;
          let thinkingTokens = 0;
          let firstChunk = true;
          
          const writeEvent = (event: string, data: object) => {
            res.write(`event: ${event}\n`);
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          };

          const messageId = `msg_${uuidv4()}`;

          let currentBlockType: 'text' | 'tool_use' | null = null;
          let currentContentIndex = -1;

          const stopCurrentBlock = () => {
            if (!currentBlockType) return;
            writeEvent('content_block_stop', {
              type: 'content_block_stop',
              index: currentContentIndex,
            });
            currentBlockType = null;
          };

          let accumulatedText = '';
          let messageStartSent = false;

          const startTextBlock = () => {
            if (currentBlockType === 'text') return;
            stopCurrentBlock();
            currentContentIndex++;
            currentBlockType = 'text';
            writeEvent('content_block_start', {
              type: 'content_block_start',
              index: currentContentIndex,
              content_block: { type: 'text', text: '' },
            });
          };

          const debugChunks: any[] = [];
          let thoughtLogged = false;

          for await (const chunkResp of streamGen) {
            // Collect chunks for debug logging
            if (process.env['DEBUG_LOG']) {
              debugChunks.push(chunkResp);
            }

            // Update usage tokens
            const usage = (chunkResp as any).usageMetadata;
            if (usage) {
                if (usage.promptTokenCount !== undefined) inputTokens = usage.promptTokenCount;
                if (usage.candidatesTokenCount !== undefined) outputTokens = usage.candidatesTokenCount;
                // Check for thinking tokens in various possible fields
                const thoughtCount = usage.thoughtsTokenCount ?? usage.thoughts_token_count ?? usage.thinking_token_count;
                if (thoughtCount !== undefined) thinkingTokens = thoughtCount;
            }

            // Send message_start on first chunk (with whatever token info we have)
            if (!messageStartSent) {
              messageStartSent = true;
              writeEvent('message_start', {
                type: 'message_start',
                message: {
                  id: messageId,
                  type: 'message',
                  role: 'assistant',
                  model,
                  content: [],
                  stop_reason: null,
                  stop_sequence: null,
                  usage: { input_tokens: inputTokens, output_tokens: 0 },
                },
              });
            }

            // Filter thought parts from chunk
            const rawParts = chunkResp.candidates?.[0]?.content?.parts || [];

            // In stream mode with Gemini Thinking, the first chunk often contains the "thought"
            // (sometimes disguised as text/system prompt) but no thoughtSignature.
            // We force log the first non-empty chunk if TEXT_LOG is on.
            const shouldForceLog = !thoughtLogged && rawParts.length > 0;
            if (shouldForceLog) thoughtLogged = true;

            logThoughtParts(rawParts, requestId, 'stream', shouldForceLog);
            const filteredParts = filterThoughtParts(rawParts);

            // Extract text from this chunk's parts
            const textParts = filteredParts.filter((p: any) => p.text && !p.functionCall);
            if (textParts.length > 0) {
              // Gemini may send cumulative or incremental text depending on model/mode, 
              // but generateContentStream usually gives incremental chunks for text.
              // However, if it acts weird, we handle it.
              // Actually, standard Gemini stream gives incremental text.
              const textDelta = textParts.map((p: any) => p.text).join('');
              
              if (textDelta.length > 0) {
                accumulatedText += textDelta;
                startTextBlock();
                writeEvent('content_block_delta', {
                  type: 'content_block_delta',
                  index: currentContentIndex,
                  delta: { type: 'text_delta', text: textDelta },
                });
              }
            }

            // Handle structured function calls
            const functionCalls = filteredParts.filter((p: any) => p.functionCall);
            if (functionCalls && functionCalls.length > 0) {
              for (const part of functionCalls) {
                if (part.functionCall) {
                  stopCurrentBlock();
                  currentContentIndex++;
                  currentBlockType = 'tool_use';
                  const toolId = `toolu_${uuidv4()}`;
                  const serializedArgs = serializeArgsForSse(part.functionCall.args ?? {});
                  writeEvent('content_block_start', {
                    type: 'content_block_start',
                    index: currentContentIndex,
                    content_block: { type: 'tool_use', id: toolId, name: part.functionCall.name, input: {} },
                  });
                  writeEvent('content_block_delta', {
                    type: 'content_block_delta',
                    index: currentContentIndex,
                    delta: { type: 'input_json_delta', partial_json: serializedArgs },
                  });
                  stopCurrentBlock();
                }
              }
            }
          }

          stopCurrentBlock();

          writeEvent('message_delta', {
            type: 'message_delta',
            delta: { stop_reason: 'end_turn', stop_sequence: null },
            usage: {
              output_tokens: outputTokens,
              thinking_tokens: thinkingTokens
            },
          });

          const totalTokens = sumTokenCounts(inputTokens, outputTokens, thinkingTokens);
          const promptStr = `prompt=${formatTokenCount(inputTokens)}`;
          const thinkingStr = ` thinking=${formatTokenCount(thinkingTokens)}`;
          const outputStr = `output=${formatTokenCount(outputTokens)}`;
          const totalStr = `total=${formatTokenCount(totalTokens)}`;
          
          logger.info(
            `[${requestId}] model=${model} usage: ${promptStr}${thinkingStr} ${outputStr} ${totalStr} tokens`,
          );

          // Send message_stop event
          res.write(`event: message_stop\n`);
          res.write(`data: ${JSON.stringify({
            type: 'message_stop',
          })}\n\n`);

          // Write debug log
          await writeDebugLog(requestId, 'stream', {
            request: {
              contents,
              config: {
                temperature,
                topP,
                maxOutputTokens,
                tools,
                systemInstruction,
              },
              model,
            },
            response: {
              chunks: debugChunks,
              accumulatedText,
              usage: {
                inputTokens,
                outputTokens,
                thinkingTokens,
                totalTokens,
              },
            },
          });

          return res.end();
        } catch (err) {
          const errorMsg = (err as Error).message || 'Stream error';
          logger.error(`[${requestId}] Stream request failed: ${errorMsg}`, err);
          if (process.env['DEBUG_LOG']) {
            await writeDebugLog(requestId, 'stream-error', {
              request: {
                contents,
                config: {
                  temperature,
                  topP,
                  maxOutputTokens,
                  tools,
                  systemInstruction,
                },
                model,
              },
              error: {
                message: errorMsg,
                name: (err as Error).name,
                stack: (err as Error).stack,
              },
            });
          }
          if (!res.headersSent) {
            res.status(502);
            res.setHeader('Content-Type', 'application/json');
          } else {
            res.statusCode = 502;
          }
          return res.end(JSON.stringify({ error: 'stream_failed', message: errorMsg }));
        } finally {
          clearInterval(keepAliveInterval);
        }
      } else {
        // Non-streaming response
        const textSizeKB = (Buffer.byteLength(JSON.stringify(contents), 'utf8') / 1024).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        logger.info(
          `[${requestId}] Sending request model=${model} stream=false text=${textSizeKB}KB`,
        );

        // Log request details if DEBUG_LOG_REQUESTS is set
        if (process.env['DEBUG_LOG_REQUESTS']) {
          try {
            const logDir = '/tmp/gemini-cli-anthropic';
            try { await fs.mkdir(logDir, { recursive: true }); } catch {}

            const logPath = path.join(logDir, `gemini-cli-anthropic-${requestId}.txt`);
            const logContent = `[INFO] ${new Date().toLocaleString()} -- [${requestId}] Sending request model=${model} stream=false text=${textSizeKB}KB\n` +
              `System: ${JSON.stringify(systemInstruction, null, 2)}\n` +
              `Tools: ${JSON.stringify(tools, null, 2)}\n` +
              `Contents:\n${JSON.stringify(contents, null, 2)}\n\n`;
            await fs.appendFile(logPath, logContent);
          } catch (e) {
            // Ignore logging errors
          }
        }

        if (process.env['DEBUG_LOG']) {
          await writeDebugLog(requestId, 'preflight', {
            model,
            stream: false,
            stats: collectDebugRequestStats(contents),
            contentDebug: summarizeContentsForDebug(contents),
            systemInstruction,
            tools,
          });
        }

        const response = await executeWithRetry(
          async () => {
            const client = config.getGeminiClient() as any;
            return client.rawGenerateContent(
              contents,
              {
                  temperature,
                  topP,
                  maxOutputTokens,
                  ...(tools && { tools: tools as any }),
                  ...(systemInstruction && { systemInstruction: systemInstruction as any }),
              },
              new AbortController().signal,
              model
            ) as Promise<any>;
          },
          requestId,
          model
        );

        const usage = (response as any).usageMetadata || (response as any).usage_metadata;
        const content: any[] = [];

        // Filter thought parts first
        const rawParts = response.candidates?.[0]?.content?.parts || [];
        logThoughtParts(rawParts, requestId, 'non-stream');
        const parts = filterThoughtParts(rawParts);

        // Extract text
        const textParts = parts.filter((p: any) => p.text && !p.functionCall);
        if (textParts.length > 0) {
          const text = textParts.map((p: any) => p.text).join('');
          content.push({ type: 'text', text });
        }

        // Check for tool calls
        for (const part of parts) {
          if ((part as any).functionCall) {
            const fc = (part as any).functionCall;
            content.push({
              type: 'tool_use',
              id: `toolu_${uuidv4()}`,
              name: fc.name,
              input: fc.args || {}
            });
          }
        }

        const prompt = usage?.promptTokenCount ?? usage?.prompt_token_count ?? 0;
        const candidates = usage?.candidatesTokenCount ?? usage?.candidates_token_count ?? 0;
        const thinking = usage?.thoughtsTokenCount ?? usage?.thoughts_token_count ?? usage?.thinking_token_count ?? 0;

        const result = {
          id: `msg_${uuidv4()}`,
          type: 'message',
          role: 'assistant',
          model,
          content,
          stop_reason: 'end_turn',
          stop_sequence: null,
          usage: {
            input_tokens: prompt,
            output_tokens: candidates,
            thinking_tokens: thinking,
          },
        };

        const totalTokens = sumTokenCounts(prompt, candidates, thinking);
        const promptStr = `prompt=${formatTokenCount(prompt)}`;
        const thinkingStr = ` thinking=${formatTokenCount(thinking)}`;
        const outputStr = `output=${formatTokenCount(candidates)}`;
        const totalStr = `total=${formatTokenCount(totalTokens)}`;
        
        logger.info(
          `[${requestId}] model=${model} usage: ${promptStr}${thinkingStr} ${outputStr} ${totalStr} tokens`,
        );

        await writeDebugLog(requestId, 'non-stream', {
          request: {
            contents,
            config: {
                temperature,
                topP,
                maxOutputTokens,
                tools,
                systemInstruction,
              },
            model,
          },
          response: {
            raw: response,
            formatted: result,
          },
        });

        return res.status(200).json(result);
      }
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : 'Bad request';

      if (res.headersSent) {
        // If headers already sent (e.g. during stream setup), we cannot send JSON error
        // Try to end the stream with an error event if possible
        logger.error(`[${requestId}] Error after headers sent: ${message}`);
        try {
          if (!res.writableEnded) {
            res.write(`event: error\n`);
            res.write(`data: ${JSON.stringify({ error: { type: 'api_error', message } })}\n\n`);
            res.end();
          }
        } catch (ignore) {
          // If socket is closed, writing might fail
        }
        return;
      }

      return res.status(400).json({
        error: {
          type: 'api_error',
          message,
        },
      });
    }
  });
}
