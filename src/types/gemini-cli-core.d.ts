declare module '@google/gemini-cli-core' {
  export const debugLogger: any;
  export const GEMINI_DIR: string;
  export const DEFAULT_GEMINI_MODEL: string;
  export const DEFAULT_GEMINI_EMBEDDING_MODEL: string;
  export function getErrorMessage(error: unknown): string;

  export type TelemetrySettings = any;
  export type TelemetryTarget = any;
  export type Config = any;
  export const Config: any;
  export type ConfigParameters = any;
  export type ExtensionLoader = any;
  export type MCPServerConfig = any;
  export type MCPServerStatus = any;
  export type ToolConfirmationOutcome = any;

  export const ApprovalMode: any;
  export const AuthType: any;
  export const FileDiscoveryService: any;
  export const loadServerHierarchicalMemory: any;

  export class SimpleExtensionLoader {
    constructor(extensions: any[]);
  }
}
