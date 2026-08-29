export type Mode = "assistant" | "story";
export type Role = "user" | "assistant";

export type Attachment = {
  id: string;
  name: string;
  mime: string;
  kind: "image" | "text";
  text?: string;
  dataUrl?: string;
  size: number;
};

export type StreamMetrics = {
  model: string;
  tokenCount: number;
  generationTime: number;
  promptTokens: number;
  tokenSavings: number;
  ttft: number;
};

export type TurnFlags = {
  agent?: boolean;
  noContext?: boolean;
  ooc?: boolean;
  includeBranch?: string;
};

export type Message = {
  id: string;
  role: Role;
  content: string;
  reasoning?: string;
  attachments?: Attachment[];
  metrics?: StreamMetrics;
  flags?: TurnFlags;
  status?: string;
  /** Quiet model label next to Processing Prompt / Streaming once picked. */
  streamingModel?: string;
  /** Orchestrator role: story, nsfw, coding, agent, … */
  streamingRoute?: string;
  /** Packed prompt size (tokens) next to Processing Prompt / Streaming. */
  streamingContext?: number;
  /** Basenames fetched via NEED_GOLD this turn. */
  recalled?: string[];
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
  locked: boolean;
  messages: Message[];
  rag: RagChunk[];
  createdAt: number;
  updatedAt: number;
};

export type ChatSnapshot = {
  currentId: string;
  branches: Record<string, Branch>;
  pendingAttachments: Attachment[];
  forceAgent: boolean;
  pendingOoc: string;
};
