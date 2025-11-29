import { GeminiClient } from '@google/gemini-cli-core';
import type { Content, GenerateContentConfig, GenerateContentResponse } from '@google/genai';

export function applyClientPatch() {
  // Monkey-patch rawGenerateContentStream to pass abortSignal
  GeminiClient.prototype.rawGenerateContentStream = async function*(
    contents: Content[],
    config: GenerateContentConfig,
    abortSignal: AbortSignal,
    model: string,
  ): AsyncGenerator<GenerateContentResponse> {
    const client = this as any;
    // Access private method
    const generator = client.getContentGeneratorOrFail();

    const stream = await generator.generateContentStream(
      {
        model,
        contents,
        config: { ...config, abortSignal },
      },
      client.lastPromptId,
    );

    for await (const chunk of stream) {
      if (abortSignal.aborted) {
        break;
      }
      yield chunk;
    }
  };

  // Monkey-patch rawGenerateContent to pass abortSignal
  GeminiClient.prototype.rawGenerateContent = async function(
    contents: Content[],
    config: GenerateContentConfig,
    abortSignal: AbortSignal,
    model: string,
  ): Promise<GenerateContentResponse> {
    const client = this as any;
    return client.getContentGeneratorOrFail().generateContent(
      {
        model,
        contents,
        config: { ...config, abortSignal },
      },
      client.lastPromptId,
    );
  };

  // No console.log here to keep output clean, or maybe debug log
}
