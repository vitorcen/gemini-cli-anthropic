/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type express from 'express';
import { v4 as uuidv4 } from 'uuid';
import type { Config } from '../gemini-cli/packages/core/src/index.js';
import { SimpleExtensionLoader } from '../gemini-cli/packages/core/src/index.js';
import { logger } from './utils/logger.js';
import fs from 'node:fs/promises';
import path from 'node:path';
import type { Content } from '@google/generative-ai';
import {
  executeWithRetry,
  formatTokenCount,
  sumTokenCounts,
  safeStringify,
  serializeArgsForSse,
  cleanSchema,
  mapModelName,
  filterThoughtParts,
  logThoughtParts,
  writeDebugLog,
  mergeConsecutiveContents,
  collectDebugRequestStats,
  summarizeContentsForDebug,
  SYNTHETIC_THOUGHT_SIGNATURE
} from './utils/proxyHelpers.js';

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

      for (let idx = 0; idx < body.messages.length; idx++) {
        const message = body.messages[idx];

        // // Skip prior model messages that explicitly said "No response requested."
        // if (
        //   message.role === 'assistant' &&
        //   typeof message.content === 'string' &&
        //   message.content.trim().toLowerCase() === 'no response requested.'
        // ) {
        //   continue;
        // }

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
              const rawContent = block.content;
              let resultPayload: { result: unknown };
              if (typeof rawContent === 'string') {
                resultPayload = { result: rawContent };
              } else {
                // Ensure the payload is JSON-serializable; fall back to plain text if not.
                try {
                  JSON.stringify(rawContent);
                  resultPayload = { result: rawContent };
                } catch {
                  logger.warn(
                    `[${requestId}] tool_result for "${resolvedName}" was not JSON-serializable; applied safeStringify fallback`,
                  );
                  resultPayload = {
                    result: safeStringify(rawContent),
                    _notice: 'tool_result was stringified because it was not JSON-serializable',
                  };
                }
              }
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
        const abortController = new AbortController();
        let clientAborted = false;
        const abortSignal = abortController.signal;
        let streamCompleted = false;

        const handleClientAbort = () => {
          if (streamCompleted) return;
          if (clientAborted) return;
          clientAborted = true;
          abortController.abort();
        };

        // Use 'aborted' as the signal that the client actively canceled.
        req.on('aborted', handleClientAbort);

        try {
          // Use ContentGenerator directly to bypass Turn logic
          const requestBytes = Buffer.byteLength(JSON.stringify(contents), 'utf8');
          const textSizeKB = (requestBytes / 1024).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
          logger.info(
            `[${requestId}] Sending request model=${model} stream=true text=${textSizeKB}KB`,
          );

          // Log request details if DEBUG_LOG is set
          if (process.env['DEBUG_LOG']) {
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
                abortSignal,
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
          let lastFinishReason: string | undefined;

          const writeEvent = (event: string, data: object) => {
            if (clientAborted || res.writableEnded) return;
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
          let receivedAnyChunk = false;
          let emittedFunctionCall = false;

          for await (const chunkResp of streamGen) {
            if (abortSignal.aborted || clientAborted) {
              if (!streamCompleted) {
                logger.info(`[${requestId}] Stream aborted by client; stopping chunk consumption.`);
              }
              stopCurrentBlock();
              // Gracefully end SSE; guard writes if client already closed.
              writeEvent('message_stop', { type: 'message_stop' });
              res.end();
              return;
            }
            receivedAnyChunk = true;
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
            const finish = (chunkResp as any)?.candidates?.[0]?.finishReason;
            if (finish) {
              lastFinishReason = finish;
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
                  emittedFunctionCall = true;
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

          const shouldFallbackDueToEmpty = !receivedAnyChunk;
          const shouldFallbackDueToTinyResponse =
             receivedAnyChunk &&
             !emittedFunctionCall &&
             accumulatedText.trim().length < 50 &&
             outputTokens <= 32 &&
             (requestBytes > 12 * 1024 || inputTokens > 512) &&
             // Only fallback if the model did NOT signal a clean stop
             lastFinishReason !== 'STOP' &&
             lastFinishReason !== 'STOPPING' &&
             lastFinishReason !== 'FINISH_REASON_UNSPECIFIED';

          const shouldFallbackDueToMissingUsage =
             receivedAnyChunk &&
             outputTokens === 0 &&
             inputTokens === 0 &&
             lastFinishReason === undefined &&
             accumulatedText.trim().length > 0 &&
             requestBytes > 12 * 1024;

          // Fallback: upstream stream returned zero chunks or tiny/early-stop text; retry once with non-stream and stream the result to avoid a second user call.
          if (shouldFallbackDueToEmpty || shouldFallbackDueToTinyResponse || shouldFallbackDueToMissingUsage) {
            const reason = shouldFallbackDueToEmpty
              ? 'no chunks'
              : shouldFallbackDueToTinyResponse
              ? `tiny response (${accumulatedText.length} chars) without clean stop`
              : 'missing usage metadata';

            logger.warn(
              `[${requestId}] Stream fallback triggered: ${reason}; falling back to non-stream in the same request`,
            );
            try {
              // Close any open block before retrying
              stopCurrentBlock();

              const fallbackResponse = await executeWithRetry(
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

              const usage = (fallbackResponse as any).usageMetadata || (fallbackResponse as any).usage_metadata;
              const prompt = usage?.promptTokenCount ?? usage?.prompt_token_count ?? 0;
              const candidates = usage?.candidatesTokenCount ?? usage?.candidates_token_count ?? 0;
              const thinking = usage?.thoughtsTokenCount ?? usage?.thoughts_token_count ?? usage?.thinking_token_count ?? 0;

              const rawParts = fallbackResponse.candidates?.[0]?.content?.parts || [];
              logThoughtParts(rawParts, requestId, 'stream-fallback');
              const parts = filterThoughtParts(rawParts);

              // If we already started the message, continue appending; otherwise start fresh.
              if (!messageStartSent) {
                messageStartSent = true;
                currentContentIndex = -1;
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
                    usage: { input_tokens: prompt, output_tokens: 0 },
                  },
                });
              }

              for (const part of parts) {
                if ((part as any).text && !(part as any).functionCall) {
                  currentContentIndex++;
                  writeEvent('content_block_start', {
                    type: 'content_block_start',
                    index: currentContentIndex,
                    content_block: { type: 'text', text: '' },
                  });
                  writeEvent('content_block_delta', {
                    type: 'content_block_delta',
                    index: currentContentIndex,
                    delta: { type: 'text_delta', text: (part as any).text },
                  });
                  writeEvent('content_block_stop', {
                    type: 'content_block_stop',
                    index: currentContentIndex,
                  });
                } else if ((part as any).functionCall) {
                  const fc = (part as any).functionCall;
                  currentContentIndex++;
                  const toolId = `toolu_${uuidv4()}`;
                  const serializedArgs = serializeArgsForSse(fc.args ?? {});
                  writeEvent('content_block_start', {
                    type: 'content_block_start',
                    index: currentContentIndex,
                    content_block: { type: 'tool_use', id: toolId, name: fc.name, input: {} },
                  });
                  writeEvent('content_block_delta', {
                    type: 'content_block_delta',
                    index: currentContentIndex,
                    delta: { type: 'input_json_delta', partial_json: serializedArgs },
                  });
                  writeEvent('content_block_stop', {
                    type: 'content_block_stop',
                    index: currentContentIndex,
                  });
                }
              }

              writeEvent('message_delta', {
                type: 'message_delta',
                delta: { stop_reason: 'end_turn', stop_sequence: null },
                usage: {
                  output_tokens: candidates,
                  thinking_tokens: thinking
                },
              });

              const totalTokens = sumTokenCounts(prompt, candidates, thinking);
              const promptStr = `prompt=${formatTokenCount(prompt)}`;
              const thinkingStr = ` thinking=${formatTokenCount(thinking)}`;
              const outputStr = `output=${formatTokenCount(candidates)}`;
              const totalStr = `total=${formatTokenCount(totalTokens)}`;

              logger.info(
                `[${requestId}] model=${model} usage: ${promptStr}${thinkingStr} ${outputStr} ${totalStr} tokens (stream fallback)`,
              );

              res.write(`event: message_stop\n`);
              res.write(`data: ${JSON.stringify({
                type: 'message_stop',
              })}\n\n`);

              if (process.env['DEBUG_LOG']) {
                await writeDebugLog(requestId, 'stream-fallback', {
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
                    raw: fallbackResponse,
                    usage: {
                      prompt,
                      candidates,
                      thinking,
                      totalTokens,
                    },
                  },
                });
              }

              return res.end();
            } finally {
              clearInterval(keepAliveInterval);
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
          streamCompleted = true;

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
          if ((abortSignal.aborted || clientAborted) && !streamCompleted) {
            logger.info(`[${requestId}] Stream request aborted by client.`);
            return;
          }

          const errorMsg = (err as Error).message || 'Stream error';
          const normalized = errorMsg.toLowerCase();
          const isQuota =
            normalized.includes('quota') ||
            normalized.includes('capacity') ||
            normalized.includes('resource has been exhausted') ||
            normalized.includes('rate') ||
            normalized.includes('429');
          if (isQuota) {
            logger.warn(
              `[${requestId}] Stream request failed (quota): ${errorMsg}`,
            );
          } else {
            logger.error(
              `[${requestId}] Stream request failed: ${errorMsg}`,
            );
          }
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
          req.off('aborted', handleClientAbort);
          clearInterval(keepAliveInterval);
        }
      } else {
        // Non-streaming response
        const textSizeKB = (Buffer.byteLength(JSON.stringify(contents), 'utf8') / 1024).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        logger.info(
          `[${requestId}] Sending request model=${model} stream=false text=${textSizeKB}KB`,
        );

        // Log request details if DEBUG_LOG is set
        if (process.env['DEBUG_LOG']) {
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
