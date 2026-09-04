export type Role = "user" | "assistant";
export type Mode = "story" | "assistant";

export type Attachment = {
  id: string;
  name: string;
  mime: string;
  kind: "image" | "text" | "file";
  text?: string;
  dataUrl?: string;
  file?: string;
};

export type StreamMetrics = {
  model?: string;
  tokenCount?: number;
  generationTime?: number;
  promptTokens?: number;
  tokenSavings?: number;
  ttft?: number;
};

export type TurnFlags = Record<string, unknown>;

export type Message = {
  id: string;
  role: Role;
  content: string;
  reasoning?: string;
  attachments?: Attachment[];
  metrics?: StreamMetrics;
  flags?: TurnFlags;
  status?: string;
  streamingModel?: string;
  streamingRoute?: string;
  streamingContext?: number;
  recalled?: string[];
  ragIds?: string[];
  ragEntryIds?: string[];
  createdAt: number;
};

export type RagChunk = {
  id: string;
  source: string;
  text: string;
};

export type Branch = {
  id: string;
  name: string;
  mode: Mode;
  locked?: boolean;
  messages: Message[];
  rag?: unknown[];
  createdAt: number;
};

export type ChatSnapshot = {
  currentId: string;
  branches: Record<string, Branch>;
};
