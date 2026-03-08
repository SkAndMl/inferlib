import styles from "./Topbar.module.css";

interface TopbarProps {
  title: string;
  modelId: string;
  thinkingEnabled: boolean;
  isGenerating: boolean;
  isSidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  onToggleThinking: () => void;
}

export default function Topbar({
  title,
  modelId,
  thinkingEnabled,
  isGenerating,
  isSidebarCollapsed,
  onToggleSidebar,
  onToggleThinking,
}: TopbarProps) {
  const modelLabel = modelId.includes("/") ? modelId.split("/").at(-1) ?? modelId : modelId;

  return (
    <header className={styles.topbar}>
      <button
        aria-label="Toggle sidebar"
        className={styles.menuButton}
        onClick={onToggleSidebar}
        type="button"
      >
        {isSidebarCollapsed ? "Show" : "Hide"}
      </button>

      <div className={styles.titleGroup}>
        <div className={styles.title}>{title}</div>
        <div className={styles.subtitle}>Serving {modelLabel}</div>
      </div>

      <div className={styles.actions}>
        <span className={styles.modelBadge}>{modelLabel}</span>
        <button
          aria-pressed={thinkingEnabled}
          className={[
            styles.thinkingButton,
            thinkingEnabled ? styles.thinkingButtonActive : "",
          ]
            .filter(Boolean)
            .join(" ")}
          disabled={isGenerating}
          onClick={onToggleThinking}
          type="button"
        >
          Thinking {thinkingEnabled ? "On" : "Off"}
        </button>
      </div>
    </header>
  );
}
