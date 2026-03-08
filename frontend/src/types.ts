export type Role = "system" | "user" | "assistant";

export interface ChatSummary {
  chat_id: string;
  title: string;
  created_at?: number;
  updated_at?: number;
  preview?: string;
}

export interface PersistedMessage {
  message_id: string;
  chat_id?: string;
  role: Role;
  content: string;
  created_at?: number;
}

export interface UiMessage {
  message_id?: string;
  role: Role;
  content: string;
  isLoading?: boolean;
  isError?: boolean;
}

export interface ChatListResponse {
  chats: ChatSummary[];
}

export interface ChatMessagesResponse {
  chat: ChatSummary;
  messages: PersistedMessage[];
}

export interface ModelListResponse {
  object: string;
  data: Array<{
    id: string;
    object: string;
    owned_by: string;
  }>;
}

export interface OpenAIMessage {
  role: Role;
  content: string;
}

export interface StreamCompletionPayload {
  model: string;
  messages: OpenAIMessage[];
  stream: boolean;
  thinking: boolean;
}
