import type { ChatSummary } from "../types";
import styles from "./Sidebar.module.css";

interface SidebarProps {
  chats: ChatSummary[];
  activeChatId: string | null;
  isGenerating: boolean;
  isMobileOpen: boolean;
  isDesktopCollapsed: boolean;
  onToggleSidebar: () => void;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
}

export default function Sidebar({
  chats,
  activeChatId,
  isGenerating,
  isMobileOpen,
  isDesktopCollapsed,
  onToggleSidebar,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}: SidebarProps) {
  return (
    <aside
      className={[
        styles.sidebar,
        isMobileOpen ? styles.mobileOpen : "",
        isDesktopCollapsed ? styles.desktopCollapsed : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div
        className={[styles.header, isDesktopCollapsed ? styles.headerCollapsed : ""]
          .filter(Boolean)
          .join(" ")}
      >
        <button
          aria-label={isDesktopCollapsed ? "Show sidebar" : "Hide sidebar"}
          className={styles.iconButton}
          onClick={onToggleSidebar}
          title={isDesktopCollapsed ? "Show sidebar" : "Hide sidebar"}
          type="button"
        >
          <span className={styles.panelIcon} aria-hidden="true" />
        </button>

        <button
          aria-label="New chat"
          className={[styles.iconButton, styles.iconButtonAccent].join(" ")}
          onClick={onNewChat}
          title="New chat"
          type="button"
        >
          <span className={styles.plusIcon} aria-hidden="true" />
        </button>
        {!isDesktopCollapsed && (
          <h1 className={styles.brand}>inferlib</h1>
        )}
      </div>

      {!isDesktopCollapsed && (
        <>
        <div className={styles.sectionLabel}>Recent conversations</div>
        <div className={styles.chatList}>
          {chats.map((chat) => {
            const isActive = chat.chat_id === activeChatId;
            const preview = chat.preview?.trim() || "No messages yet";
            return (
              <div className={styles.chatRow} key={chat.chat_id}>
                <button
                  className={[styles.chatItem, isActive ? styles.chatItemActive : ""]
                    .filter(Boolean)
                    .join(" ")}
                  onClick={() => onSelectChat(chat.chat_id)}
                  type="button"
                >
                  <div className={styles.chatTitle}>{chat.title || "New chat"}</div>
                  <div className={styles.chatPreview}>{preview}</div>
                </button>

                <button
                  aria-label="Delete chat"
                  className={styles.deleteButton}
                  disabled={isGenerating && isActive}
                  onClick={() => onDeleteChat(chat.chat_id)}
                  title="Delete chat"
                  type="button"
                >
                  x
                </button>
              </div>
            );
          })}
        </div>
        </>
      )}
    </aside>
  );
}
