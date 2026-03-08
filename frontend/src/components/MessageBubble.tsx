import { renderMarkdown, splitThinkingSegments } from "../markdown";
import type { UiMessage } from "../types";
import styles from "./MessageBubble.module.css";

interface MessageBubbleProps {
  message: UiMessage;
  messageKey: string;
  openThinkingPanels: Record<number, boolean> | undefined;
  onToggleThinkingPanel: (messageKey: string, index: number, isOpen: boolean) => void;
}

function LoadingIndicator() {
  return (
    <div className={styles.loading}>
      <span />
      <span />
      <span />
    </div>
  );
}

export default function MessageBubble({
  message,
  messageKey,
  openThinkingPanels,
  onToggleThinkingPanel,
}: MessageBubbleProps) {
  const isAssistant = message.role === "assistant" && !message.isError;
  const segments = isAssistant ? splitThinkingSegments(message.content) : [];
  const hasThinking = segments.some((segment) => segment.type === "thinking");

  return (
    <div
      className={[
        styles.row,
        message.role === "user" ? styles.userRow : "",
        message.isError ? styles.errorRow : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div
        className={[
          styles.message,
          message.role === "user" ? styles.userMessage : "",
          isAssistant ? styles.assistantMessage : "",
          message.isError ? styles.errorMessage : "",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        {message.isLoading ? (
          <LoadingIndicator />
        ) : isAssistant ? (
          <>
            {!hasThinking && (
              <div
                className={styles.richText}
                dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
              />
            )}

            {hasThinking &&
              segments.map((segment, index) => {
                if (segment.type === "text") {
                  if (!segment.content.trim()) {
                    return null;
                  }
                  return (
                    <div
                      className={styles.richText}
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(segment.content) }}
                      key={`${messageKey}-text-${index}`}
                    />
                  );
                }

                const isOpen =
                  typeof openThinkingPanels?.[index] === "boolean"
                    ? openThinkingPanels[index]
                    : Boolean(segment.inProgress);
                const summary = segment.inProgress ? "Thinking (in progress)" : "Thinking";

                return (
                  <details
                    className={[
                      styles.thinkingBlock,
                      segment.inProgress ? styles.thinkingBlockOpen : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    key={`${messageKey}-thinking-${index}`}
                    onToggle={(event) =>
                      onToggleThinkingPanel(
                        messageKey,
                        index,
                        (event.currentTarget as HTMLDetailsElement).open
                      )
                    }
                    open={isOpen}
                  >
                    <summary>{summary}</summary>
                    <div
                      className={[styles.richText, styles.thinkingContent].join(" ")}
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(segment.content) }}
                    />
                  </details>
                );
              })}
          </>
        ) : (
          <div className={styles.plainText}>{message.content}</div>
        )}
      </div>
    </div>
  );
}
