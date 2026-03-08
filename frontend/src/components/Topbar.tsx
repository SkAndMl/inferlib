import styles from "./Topbar.module.css";

interface TopbarProps {
  title: string;
  modelId: string;
  thinkingEnabled: boolean;
  isGenerating: boolean;
  onToggleThinking: () => void;
}

export default function Topbar({
  title,
  modelId,
  thinkingEnabled,
  isGenerating,
  onToggleThinking,
}: TopbarProps) {
  const modelLabel = modelId.includes("/") ? modelId.split("/").at(-1) ?? modelId : modelId;

  return (
    <header className={styles.topbar}>
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
