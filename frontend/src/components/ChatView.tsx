import MessageBubble from "./MessageBubble";
import styles from "./ChatView.module.css";
import type { UiMessage } from "../types";

interface ChatViewProps {
  activeChatId: string | null;
  messages: UiMessage[];
  thinkingPanels: Record<string, Record<number, boolean>>;
  onToggleThinkingPanel: (messageKey: string, index: number, isOpen: boolean) => void;
}

function getMessageKey(message: UiMessage, index: number, activeChatId: string | null): string {
  if (message.message_id) {
    return message.message_id;
  }
  return `${activeChatId ?? "chat"}:${index}`;
}

export default function ChatView({
  activeChatId,
  messages,
  thinkingPanels,
  onToggleThinkingPanel,
}: ChatViewProps) {
  if (!messages.length) {
    return (
      <div className={styles.emptyState}>
        <p className={styles.emptyEyebrow}>Ready when you are</p>
        <h2 className={styles.emptyTitle}>Ask a question, sketch an idea, or probe the model.</h2>
        <p className={styles.emptyBody}>
          inferlib keeps the chat local while the interface stays focused and quiet.
        </p>
      </div>
    );
  }

  return (
    <div className={styles.stream}>
      {messages.map((message, index) => {
        const messageKey = getMessageKey(message, index, activeChatId);
        return (
          <MessageBubble
            key={messageKey}
            message={message}
            messageKey={messageKey}
            onToggleThinkingPanel={onToggleThinkingPanel}
            openThinkingPanels={thinkingPanels[messageKey]}
          />
        );
      })}
    </div>
  );
}
