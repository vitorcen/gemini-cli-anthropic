/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, test, expect, vi, beforeEach, afterEach } from 'vitest';
import express from 'express';
import request from 'supertest';
import { registerClaudeEndpoints } from './claudeProxy';

// --- Mocks ---

const mockGenerateContentStream = vi.fn();
const mockGenerateContent = vi.fn();

// Mock the Config object structure
const mockGetContentGenerator = vi.fn(() => ({
  generateContentStream: mockGenerateContentStream,
  generateContent: mockGenerateContent,
}));

const mockInitialize = vi.fn();
const mockSetSystemInstruction = vi.fn();
const mockSetTools = vi.fn();
const mockGetChat = vi.fn(() => ({
  setSystemInstruction: mockSetSystemInstruction,
  setTools: mockSetTools,
}));

const mockGetGeminiClient = vi.fn(() => ({
  initialize: mockInitialize,
  isInitialized: () => true,
  getChat: mockGetChat,
}));

const mockConfig = {
  getContentGenerator: mockGetContentGenerator,
  getGeminiClient: mockGetGeminiClient,
};

// Mock module dependencies
vi.mock('@google/gemini-cli-core', () => ({
  DEFAULT_GEMINI_FLASH_MODEL: 'gemini-2.5-flash',
  DEFAULT_GEMINI_MODEL: 'gemini-2.5-pro',
  SimpleExtensionLoader: class {},
  GeminiEventType: {
    Content: 'content',
    ToolCallRequest: 'tool_call_request',
    Finished: 'finished',
    Error: 'error'
  }
}));

vi.mock('@a2a/config/config.js', () => ({
  loadConfig: vi.fn().mockResolvedValue(mockConfig),
}));

vi.mock('@a2a/config/settings.js', () => ({
  loadSettings: vi.fn().mockReturnValue({}),
}));

// Mock logger to reduce noise during tests
vi.mock('./utils/logger', () => ({
  logger: {
    info: vi.fn(),
    debug: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
  },
}));

// --- Test Suite ---

describe('Claude Proxy (gemini-cli-anthropic)', () => {
  let app: express.Express;

  beforeEach(() => {
    vi.clearAllMocks();
    app = express();
    app.use(express.json({ limit: '50mb' })); // Match index.ts
    const router = express.Router();
    registerClaudeEndpoints(router, mockConfig as any);
    app.use('/v1', router);
  });

  // 1. Basic Chat (Non-Streaming)
  // Replaces: "should handle a non-streaming chat message"
  test('Chat (Non-Streaming): should return correct response format', async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ text: 'Hello world' }] } }],
      usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5 }
    });

    const response = await request(app)
      .post('/v1/messages')
      .send({
        model: 'gemini-2.5-flash',
        messages: [{ role: 'user', content: 'Hi' }],
        max_tokens: 100
      });

    expect(response.status).toBe(200);
    expect(response.body).toEqual({
      id: expect.stringMatching(/^msg_/),
      type: 'message',
      role: 'assistant',
      model: 'gemini-2.5-flash',
      content: [{ type: 'text', text: 'Hello world' }],
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 5 }
    });

    // Verify underlying call
    expect(mockGenerateContent).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'gemini-2.5-flash',
        contents: [{ role: 'user', parts: [{ text: 'Hi' }] }]
      }),
      expect.any(String) // requestId
    );
  });

  // 2. Basic Chat (Streaming)
  // Replaces: "should handle a streaming chat message"
  test('Chat (Streaming): should emit SSE events correctly', async () => {
    async function* mockStream() {
      yield {
        candidates: [{ content: { parts: [{ text: 'Hello' }] } }],
        usageMetadata: { promptTokenCount: 5 }
      };
      yield {
        candidates: [{ content: { parts: [{ text: ' World' }] } }],
        usageMetadata: { candidatesTokenCount: 2 }
      };
    }
    mockGenerateContentStream.mockResolvedValue(mockStream());

    const response = await request(app)
      .post('/v1/messages')
      .send({
        model: 'gemini-2.5-flash',
        messages: [{ role: 'user', content: 'Hi' }],
        stream: true
      });

    expect(response.status).toBe(200);
    expect(response.headers['content-type']).toBe('text/event-stream');

    const text = response.text;
    
    // Verify Event Sequence
    expect(text).toContain('event: message_start');
    expect(text).toContain('"type":"message_start"');
    
    expect(text).toContain('event: content_block_start');
    expect(text).toContain('"type":"content_block_start"');
    
    expect(text).toContain('event: content_block_delta');
    expect(text).toContain('"text":"Hello"');
    expect(text).toContain('"text":" World"');
    
    expect(text).toContain('event: message_delta');
    expect(text).toContain('"stop_reason":"end_turn"');
    
    expect(text).toContain('event: message_stop');
  });

  // 3. History Sanitization (Critical Fix)
  // Covers: "Sanitization: should convert historical tool_use to [SYSTEM LOG] text"
  test('History Sanitization: should emit functionCall/functionResponse for tool history', async () => {
    mockGenerateContent.mockResolvedValue({
      candidates: [{ content: { parts: [{ text: 'Acknowledged' }] } }]
    });

    // Simulate history: User -> Assistant (Tool Use) -> User (Tool Result)
    await request(app)
      .post('/v1/messages')
      .send({
        model: 'gemini-2.5-flash',
        messages: [
          { role: 'user', content: 'List files' },
          {
            role: 'assistant', 
            content: [
              { type: 'text', text: 'I will list files.' },
              { type: 'tool_use', id: 'call_1', name: 'ls', input: { path: '.' } }
            ] 
          },
          {
            role: 'user', 
            content: [
              { type: 'tool_result', tool_use_id: 'call_1', content: 'file1.txt\nfile2.txt' }
            ] 
          }
        ]
      });

    const callArgs = mockGenerateContent.mock.calls[0][0];
    const contents = callArgs.contents;

    expect(contents).toHaveLength(3);

    // 1. User normal message
    expect(contents[0].parts[0].text).toBe('List files');

    // 2. Assistant message (Sanitized)
    expect(contents[1].role).toBe('model');
    // Should keep text part
    expect(contents[1].parts[0].text).toBe('I will list files.');
    const toolUsePart = contents[1].parts[1].functionCall;
    expect(toolUsePart).toEqual({
      name: 'ls',
      args: { path: '.' }
    });

    // 3. User message (Sanitized Result)
    expect(contents[2].role).toBe('user');
    const toolResultPart = contents[2].parts[0].functionResponse;
    expect(toolResultPart).toEqual({
      name: 'ls',
      response: { result: 'file1.txt\nfile2.txt' }
    });
  });

  // 3.1 Sequential Tool Use (Multi-tool Sanitization)
  test('History Sanitization: should handle sequential tool uses in one turn', async () => {
    mockGenerateContent.mockResolvedValue({ candidates: [] });

    await request(app)
      .post('/v1/messages')
      .send({
        model: 'gemini-2.5-flash',
        messages: [
          { role: 'user', content: 'Check files' },
          { 
            role: 'assistant', 
            content: [
              { type: 'text', text: 'I will run two commands.' },
              { type: 'tool_use', id: 'call_1', name: 'ls', input: { path: '.' } },
              { type: 'tool_use', id: 'call_2', name: 'grep', input: { pattern: 'foo' } }
            ] 
          },
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_1', content: 'file1' },
              { type: 'tool_result', tool_use_id: 'call_2', content: 'match' }
            ]
          }
        ]
      });

    const callArgs = mockGenerateContent.mock.calls[0][0];
    const contents = callArgs.contents;
    
    // Assistant message should have 3 parts: Text, ToolCall 1, ToolCall 2
    const assistantMsg = contents[1];
    expect(assistantMsg.parts).toHaveLength(3);
    expect(assistantMsg.parts[0].text).toBe('I will run two commands.');
    expect(assistantMsg.parts[1].functionCall).toEqual({
      name: 'ls',
      args: { path: '.' }
    });
    expect(assistantMsg.parts[2].functionCall).toEqual({
      name: 'grep',
      args: { pattern: 'foo' }
    });

    // User message should have 2 parts: Result 1, Result 2
    const userMsg = contents[2];
    expect(userMsg.parts).toHaveLength(2);
    expect(userMsg.parts[0].functionResponse).toEqual({
      name: 'ls',
      response: { result: 'file1' }
    });
    expect(userMsg.parts[1].functionResponse).toEqual({
      name: 'grep',
      response: { result: 'match' }
    });
  });

  test('History Sanitization: should handle tool_id-only references', async () => {
    mockGenerateContent.mockResolvedValue({ candidates: [] });

    await request(app)
      .post('/v1/messages')
      .send({
        model: 'gemini-2.5-flash',
        messages: [
          { role: 'user', content: 'Check files again' },
          {
            role: 'assistant',
            content: [
              { type: 'text', text: 'I will run read command.' },
              { type: 'tool_use', tool_id: 'read_123', name: 'Read', input: { path: '.' } }
            ]
          },
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_id: 'read_123', content: 'contents' }
            ]
          }
        ]
      });

    const callArgs = mockGenerateContent.mock.calls[0][0];
    const sanitizedAssistant = callArgs.contents[1];
    expect(sanitizedAssistant.parts[1].functionCall).toEqual({
      name: 'Read',
      args: { path: '.' }
    });

    const sanitizedUser = callArgs.contents[2];
    expect(sanitizedUser.parts[0].functionResponse).toEqual({
      name: 'Read',
      response: { result: 'contents' }
    });
  });

  // 4. Tool Use Generation (Model Output)
  // Replaces: "should handle a streaming message with a tool call"
  test('Tool Generation: should handle model generating a tool call (Streaming)', async () => {
    async function* mockToolStream() {
      yield {
        candidates: [{ 
          content: { 
            parts: [{ 
              functionCall: { name: 'get_weather', args: { city: 'Tokyo' } } 
            }] 
          } 
        }],
        usageMetadata: { promptTokenCount: 10 }
      };
    }
    mockGenerateContentStream.mockResolvedValue(mockToolStream());

    const response = await request(app)
      .post('/v1/messages')
      .send({
        model: 'gemini-2.5-flash',
        messages: [{ role: 'user', content: 'Weather in Tokyo?' }],
        tools: [{ name: 'get_weather', input_schema: { type: 'object' } }],
        stream: true
      });

    const text = response.text;
    
    // Verify Tool Call Events
    expect(text).toContain('event: content_block_start');
    expect(text).toContain('"type":"tool_use"');
    expect(text).toContain('"name":"get_weather"');
    
    expect(text).toContain('event: content_block_delta');
    expect(text).toContain('"type":"input_json_delta"');
    expect(text).toContain('Tokyo');
  });

  test('Tool Generation: should emit {} args when functionCall has no args', async () => {
    async function* mockToolStream() {
      yield {
        candidates: [{ content: { parts: [{ functionCall: { name: 'noop' } }] } }],
      };
    }
    mockGenerateContentStream.mockResolvedValue(mockToolStream());

    const response = await request(app)
      .post('/v1/messages')
      .send({
        model: 'gemini-2.5-flash',
        messages: [{ role: 'user', content: 'Call noop' }],
        tools: [{ name: 'noop', input_schema: { type: 'object' } }],
        stream: true
      });

    expect(response.text).toContain('"partial_json":"{}"');
  });

  // 5. Retry Logic (Quota Exhausted)
  // Covers: "Retry Logic: should retry on quota exhausted error"
  test('Retry Logic: should wait and retry on quota exhausted error', async () => {
    // First call fails
    mockGenerateContent.mockRejectedValueOnce(new Error('API Error: 400 ... You have exhausted your capacity ... reset after 0s.'));
    // Second call succeeds
    mockGenerateContent.mockResolvedValueOnce({
      candidates: [{ content: { parts: [{ text: 'Success after retry' }] } }]
    });

    const start = Date.now();
    const response = await request(app)
      .post('/v1/messages')
      .send({
        model: 'gemini-2.5-flash',
        messages: [{ role: 'user', content: 'Retry me' }]
      });
    const duration = Date.now() - start;

    expect(response.status).toBe(200);
    expect(response.body.content[0].text).toBe('Success after retry');
    expect(mockGenerateContent).toHaveBeenCalledTimes(2);
    
    // Should have waited at least 1000ms (reset 0s + 1s buffer)
    expect(duration).toBeGreaterThanOrEqual(1000);
  });

  // 6. System Instruction Injection (Verification of REMOVAL)
  // Covers: "System Instruction: should NOT inject reminder"
  test('System Instruction: should NOT inject artificial reminders', async () => {
    mockGenerateContent.mockResolvedValue({ candidates: [] });

    await request(app)
      .post('/v1/messages')
      .send({
        model: 'gemini-2.5-flash',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [{ name: 'test', input_schema: {} }],
        system: 'User defined system prompt'
      });

    const callArgs = mockGenerateContent.mock.calls[0][0];
    const sys = callArgs.config.systemInstruction;
    
    // Should be exactly what user sent, no "IMPORTANT" appended
    expect(sys.parts[0].text).toBe('User defined system prompt');
    expect(sys.parts[0].text).not.toContain('IMPORTANT');
    expect(sys.parts[0].text).not.toContain('[SYSTEM LOG]');
  });

});
