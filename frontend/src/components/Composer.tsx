import type { KeyboardEvent, RefObject } from "react";
import styles from "./Composer.module.css";

interface ComposerProps {
  inputRef: RefObject<HTMLTextAreaElement | null>;
  isGenerating: boolean;
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
}

export default function Composer({
  inputRef,
  isGenerating,
  value,
  onChange,
  onSend,
}: ComposerProps) {
  const isDisabled = isGenerating || !value.trim();

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  }

  return (
    <div className={styles.shell}>
      <div className={styles.composer}>
        <textarea
          className={styles.input}
          disabled={isGenerating}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about the model, math, code, or architecture"
          ref={inputRef}
          rows={1}
          value={value}
        />
        <button
          className={styles.sendButton}
          disabled={isDisabled}
          onClick={onSend}
          type="button"
        >
          Send
        </button>
      </div>
      <p className={styles.caption}>Enter to send. Shift + Enter for a new line.</p>
    </div>
  );
}
