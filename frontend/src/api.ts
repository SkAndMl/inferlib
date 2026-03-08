import type {
  ChatListResponse,
  ChatMessagesResponse,
  ModelListResponse,
  StreamCompletionPayload,
} from "./types";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function listChats(): Promise<ChatListResponse> {
  const response = await fetch("/v1/chats");
  return parseJson<ChatListResponse>(response);
}

export async function getChatMessages(chatId: string): Promise<ChatMessagesResponse> {
  const response = await fetch(`/v1/chats/${encodeURIComponent(chatId)}/messages`);
  if (response.status === 404) {
    throw new Error("CHAT_NOT_FOUND");
  }
  return parseJson<ChatMessagesResponse>(response);
}

export async function saveMessage(
  chatId: string,
  role: "system" | "user" | "assistant",
  content: string,
  messageId: string
): Promise<void> {
  const response = await fetch(`/v1/chats/${encodeURIComponent(chatId)}/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      role,
      content,
      message_id: messageId,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to save message: ${response.status}`);
  }
}

export async function deleteChat(chatId: string): Promise<void> {
  const response = await fetch(`/v1/chats/${encodeURIComponent(chatId)}`, {
    method: "DELETE",
  });

  if (!response.ok && response.status !== 404) {
    throw new Error(`Failed to delete chat: ${response.status}`);
  }
}

export async function listModels(): Promise<ModelListResponse> {
  const response = await fetch("/v1/models");
  return parseJson<ModelListResponse>(response);
}

function consumeSseEvent(eventText: string, onChunk: (chunk: string) => void): void {
  const lines = eventText.split(/\r?\n/);
  for (const line of lines) {
    if (!line.startsWith("data:")) {
      continue;
    }

    const raw = line.slice(5).trim();
    if (!raw || raw === "[DONE]") {
      continue;
    }

    const payload = JSON.parse(raw) as {
      choices?: Array<{
        delta?: {
          content?: string;
        };
      }>;
    };
    const chunk = payload.choices?.[0]?.delta?.content;
    if (typeof chunk === "string" && chunk.length > 0) {
      onChunk(chunk);
    }
  }
}

export async function streamCompletion(
  payload: StreamCompletionPayload,
  onChunk: (chunk: string) => void
): Promise<void> {
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP error ${response.status}: ${errorText}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.toLowerCase().startsWith("text/event-stream")) {
    const rawBody = await response.text();
    if (!rawBody.trim()) {
      return;
    }

    try {
      const parsed = JSON.parse(rawBody) as {
        choices?: Array<{
          message?: {
            content?: string;
          };
        }>;
      };
      const content = parsed.choices?.[0]?.message?.content;
      if (typeof content === "string" && content.length > 0) {
        onChunk(content);
      }
      return;
    } catch {
      onChunk(rawBody);
      return;
    }
  }

  if (!response.body) {
    throw new Error("Readable stream not available in response");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() ?? "";

    for (const eventText of events) {
      consumeSseEvent(eventText, onChunk);
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    consumeSseEvent(buffer, onChunk);
  }
}
