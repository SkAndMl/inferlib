import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  deleteChat as deleteChatRequest,
  getChatMessages,
  listChats,
  listModels,
  saveMessage,
  streamCompletion,
} from "./api";
import ChatView from "./components/ChatView";
import Composer from "./components/Composer";
import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";
import styles from "./App.module.css";
import type { ChatSummary, OpenAIMessage, UiMessage } from "./types";

const DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B";
const SIDEBAR_COLLAPSED_KEY = "inferlib_sidebar_collapsed";

function uniqueId(prefix = "id"): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

function deriveTitle(content: string): string {
  const value = String(content || "").trim();
  if (!value) {
    return "New chat";
  }
  return value.split(/\s+/).slice(0, 4).join(" ");
}

function getCombinedChats(chats: ChatSummary[], draftChats: ChatSummary[]): ChatSummary[] {
  const persistedIds = new Set(chats.map((chat) => chat.chat_id));
  const unsaved = draftChats.filter((chat) => !persistedIds.has(chat.chat_id));
  return [...unsaved, ...chats].sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0));
}

function toOpenAIMessages(messages: UiMessage[]): OpenAIMessage[] {
  return messages
    .filter(
      (message) =>
        !message.isLoading &&
        !message.isError &&
        typeof message.content === "string" &&
        ["system", "user", "assistant"].includes(message.role)
    )
    .map((message) => ({
      role: message.role,
      content: message.content,
    }));
}

function isDesktopViewport(): boolean {
  return typeof window !== "undefined" ? window.innerWidth > 960 : true;
}

function readSidebarPreference(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
}

export default function App() {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [draftChats, setDraftChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [activeMessages, setActiveMessages] = useState<UiMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [thinkingEnabled, setThinkingEnabled] = useState(false);
  const [thinkingPanels, setThinkingPanels] = useState<Record<string, Record<number, boolean>>>(
    {}
  );
  const [isDesktopSidebarCollapsed, setIsDesktopSidebarCollapsed] = useState(
    readSidebarPreference()
  );
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [modelId, setModelId] = useState(DEFAULT_MODEL_ID);

  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const chatsRef = useRef(chats);
  const draftChatsRef = useRef(draftChats);
  const activeChatIdRef = useRef(activeChatId);
  const activeMessagesRef = useRef(activeMessages);
  const isGeneratingRef = useRef(isGenerating);

  useEffect(() => {
    chatsRef.current = chats;
  }, [chats]);

  useEffect(() => {
    draftChatsRef.current = draftChats;
  }, [draftChats]);

  useEffect(() => {
    activeChatIdRef.current = activeChatId;
  }, [activeChatId]);

  useEffect(() => {
    activeMessagesRef.current = activeMessages;
  }, [activeMessages]);

  useEffect(() => {
    isGeneratingRef.current = isGenerating;
  }, [isGenerating]);

  useEffect(() => {
    if (!inputRef.current) {
      return;
    }
    inputRef.current.style.height = "auto";
    inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 220)}px`;
  }, [inputValue]);

  useEffect(() => {
    if (!scrollRef.current) {
      return;
    }
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [activeMessages]);

  useEffect(() => {
    async function loadModelId() {
      try {
        const response = await listModels();
        const firstModel = response.data?.[0]?.id;
        if (firstModel) {
          setModelId(firstModel);
        }
      } catch {
        setModelId(DEFAULT_MODEL_ID);
      }
    }

    void loadModelId();
  }, []);

  useEffect(() => {
    function handleResize() {
      if (isDesktopViewport()) {
        setIsMobileSidebarOpen(false);
      }
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsMobileSidebarOpen(false);
      }
    }

    window.addEventListener("resize", handleResize);
    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("keydown", handleEscape);
    };
  }, []);

  useEffect(() => {
    async function init() {
      try {
        await loadChatsIntoState();
        if (activeChatIdRef.current) {
          await selectChat(activeChatIdRef.current);
        }
      } catch (error) {
        console.error(error);
        const fallbackMessages = [
          {
            role: "assistant" as const,
            content: "Failed to load chats. Please refresh the page.",
            isError: true,
          },
        ];
        activeMessagesRef.current = fallbackMessages;
        setActiveMessages(fallbackMessages);
      }

      inputRef.current?.focus();
    }

    void init();
  }, []);

  const combinedChats = useMemo(() => getCombinedChats(chats, draftChats), [chats, draftChats]);

  const activeTitle =
    combinedChats.find((chat) => chat.chat_id === activeChatId)?.title || "New chat";

  async function loadChatsIntoState(): Promise<void> {
    const data = await listChats();
    const nextChats = Array.isArray(data.chats) ? data.chats : [];
    const persistedIds = new Set(nextChats.map((chat) => chat.chat_id));
    const nextDraftChats = draftChatsRef.current.filter((chat) => !persistedIds.has(chat.chat_id));
    const nextCombinedChats = getCombinedChats(nextChats, nextDraftChats);

    chatsRef.current = nextChats;
    draftChatsRef.current = nextDraftChats;
    setChats(nextChats);
    setDraftChats(nextDraftChats);

    if (
      !activeChatIdRef.current ||
      !nextCombinedChats.some((chat) => chat.chat_id === activeChatIdRef.current)
    ) {
      if (nextCombinedChats.length) {
        activeChatIdRef.current = nextCombinedChats[0].chat_id;
        setActiveChatId(nextCombinedChats[0].chat_id);
      } else {
        createDraftChat();
      }
    }
  }

  async function loadMessages(chatId: string): Promise<void> {
    try {
      const data = await getChatMessages(chatId);
      const nextMessages = Array.isArray(data.messages)
        ? data.messages.map((message) => ({
            role: message.role,
            content: message.content,
            message_id: message.message_id,
          }))
        : [];
      activeMessagesRef.current = nextMessages;
      setActiveMessages(nextMessages);
    } catch (error) {
      if (error instanceof Error && error.message === "CHAT_NOT_FOUND") {
        activeMessagesRef.current = [];
        setActiveMessages([]);
        return;
      }
      throw error;
    }
  }

  async function selectChat(chatId: string): Promise<void> {
    activeChatIdRef.current = chatId;
    setActiveChatId(chatId);
    const isDraft =
      draftChatsRef.current.some((chat) => chat.chat_id === chatId) &&
      !chatsRef.current.some((chat) => chat.chat_id === chatId);

    if (isDraft) {
      activeMessagesRef.current = [];
      setActiveMessages([]);
      return;
    }

    await loadMessages(chatId);
  }

  function createDraftChat(): string {
    const chatId = uniqueId("chat");
    const now = Math.floor(Date.now() / 1000);
    const draft: ChatSummary = {
      chat_id: chatId,
      title: "New chat",
      preview: "",
      updated_at: now,
    };

    const nextDraftChats = [draft, ...draftChatsRef.current];
    draftChatsRef.current = nextDraftChats;
    activeChatIdRef.current = chatId;
    activeMessagesRef.current = [];

    setDraftChats(nextDraftChats);
    setActiveChatId(chatId);
    setActiveMessages([]);
    return chatId;
  }

  function updateChatFromMessage(chatId: string, content: string): void {
    const now = Math.floor(Date.now() / 1000);
    const title = deriveTitle(content);
    let found = false;

    const nextChats = chatsRef.current.map((chat) => {
      if (chat.chat_id !== chatId) {
        return chat;
      }
      found = true;
      return {
        ...chat,
        updated_at: now,
        preview: content,
        title: !chat.title || chat.title === "New chat" ? title : chat.title,
      };
    });

    const nextDraftChats = draftChatsRef.current.map((chat) => {
      if (chat.chat_id !== chatId) {
        return chat;
      }
      found = true;
      return {
        ...chat,
        updated_at: now,
        preview: content,
        title: !chat.title || chat.title === "New chat" ? title : chat.title,
      };
    });

    if (!found) {
      nextDraftChats.unshift({
        chat_id: chatId,
        title,
        preview: content,
        updated_at: now,
      });
    }

    chatsRef.current = nextChats;
    draftChatsRef.current = nextDraftChats;
    setChats(nextChats);
    setDraftChats(nextDraftChats);
  }

  async function handleDeleteChat(chatId: string): Promise<void> {
    if (isGeneratingRef.current && chatId === activeChatIdRef.current) {
      return;
    }

    const ok = window.confirm("Delete this chat permanently?");
    if (!ok) {
      return;
    }

    const isPersisted = chatsRef.current.some((chat) => chat.chat_id === chatId);
    if (isPersisted) {
      await deleteChatRequest(chatId);
    }

    const nextChats = chatsRef.current.filter((chat) => chat.chat_id !== chatId);
    const nextDraftChats = draftChatsRef.current.filter((chat) => chat.chat_id !== chatId);

    chatsRef.current = nextChats;
    draftChatsRef.current = nextDraftChats;
    setChats(nextChats);
    setDraftChats(nextDraftChats);

    if (activeChatIdRef.current === chatId) {
      const remainingChats = getCombinedChats(nextChats, nextDraftChats);
      if (remainingChats.length) {
        await selectChat(remainingChats[0].chat_id);
      } else {
        createDraftChat();
      }
    }
  }

  function closeMobileSidebar(): void {
    if (!isDesktopViewport()) {
      setIsMobileSidebarOpen(false);
    }
  }

  function focusComposer(): void {
    window.requestAnimationFrame(() => inputRef.current?.focus());
  }

  async function handleSendMessage(): Promise<void> {
    if (isGeneratingRef.current) {
      return;
    }

    const text = inputValue.trim();
    if (!text) {
      return;
    }

    const chatId = activeChatIdRef.current ?? createDraftChat();
    const userMessageId = uniqueId("msg");
    const assistantMessageId = uniqueId("msg");
    const nextMessages: UiMessage[] = [
      ...activeMessagesRef.current,
      { role: "user", content: text, message_id: userMessageId },
      { role: "assistant", content: "", isLoading: true, message_id: assistantMessageId },
    ];

    setInputValue("");
    setIsGenerating(true);
    isGeneratingRef.current = true;
    closeMobileSidebar();

    activeMessagesRef.current = nextMessages;
    setActiveMessages(nextMessages);
    updateChatFromMessage(chatId, text);

    const assistantIndex = nextMessages.length - 1;
    let assistantText = "";

    try {
      await saveMessage(chatId, "user", text, userMessageId);
      await streamCompletion(
        {
          model: modelId,
          messages: toOpenAIMessages(nextMessages),
          stream: true,
          thinking: thinkingEnabled,
        },
        (chunk) => {
          assistantText += chunk;
          const updatedMessages = [...activeMessagesRef.current];
          updatedMessages[assistantIndex] = {
            role: "assistant",
            content: assistantText,
            isLoading: false,
            message_id: assistantMessageId,
          };
          activeMessagesRef.current = updatedMessages;
          setActiveMessages(updatedMessages);
        }
      );

      if (!assistantText) {
        const updatedMessages = [...activeMessagesRef.current];
        updatedMessages[assistantIndex] = {
          role: "assistant",
          content: "",
          isLoading: false,
          message_id: assistantMessageId,
        };
        activeMessagesRef.current = updatedMessages;
        setActiveMessages(updatedMessages);
      } else {
        await saveMessage(chatId, "assistant", assistantText, assistantMessageId);
      }

      await loadChatsIntoState();
    } catch (error) {
      console.error(error);
      const updatedMessages = [...activeMessagesRef.current];
      updatedMessages[assistantIndex] = {
        role: "assistant",
        content: "Failed to generate response. Please try again.",
        isError: true,
        message_id: assistantMessageId,
      };
      activeMessagesRef.current = updatedMessages;
      setActiveMessages(updatedMessages);
    } finally {
      setIsGenerating(false);
      isGeneratingRef.current = false;
      focusComposer();
    }
  }

  function handleNewChat(): void {
    createDraftChat();
    closeMobileSidebar();
    focusComposer();
  }

  function handleToggleSidebar(): void {
    if (!isDesktopViewport()) {
      setIsMobileSidebarOpen((current) => !current);
      return;
    }

    const nextValue = !isDesktopSidebarCollapsed;
    setIsDesktopSidebarCollapsed(nextValue);
    window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, nextValue ? "1" : "0");
  }

  function handleToggleThinkingPanel(messageKey: string, index: number, isOpen: boolean): void {
    setThinkingPanels((current) => ({
      ...current,
      [messageKey]: {
        ...(current[messageKey] || {}),
        [index]: isOpen,
      },
    }));
  }

  const shellStyle = {
    "--sidebar-width": isDesktopSidebarCollapsed ? "72px" : "280px",
  } as CSSProperties;

  return (
    <div className={styles.shell} style={shellStyle}>
      <Sidebar
        activeChatId={activeChatId}
        chats={combinedChats}
        isDesktopCollapsed={isDesktopSidebarCollapsed}
        isGenerating={isGenerating}
        isMobileOpen={isMobileSidebarOpen}
        onToggleSidebar={handleToggleSidebar}
        onDeleteChat={(chatId) => {
          void handleDeleteChat(chatId);
        }}
        onNewChat={handleNewChat}
        onSelectChat={(chatId) => {
          void selectChat(chatId);
          closeMobileSidebar();
        }}
      />

      <button
        aria-label="Close sidebar"
        className={[
          styles.backdrop,
          isMobileSidebarOpen ? styles.backdropVisible : "",
        ]
          .filter(Boolean)
          .join(" ")}
        onClick={closeMobileSidebar}
        type="button"
      />

      <main className={styles.main}>
        <Topbar
          isGenerating={isGenerating}
          isMobileSidebarOpen={isMobileSidebarOpen}
          modelId={modelId}
          onToggleSidebar={handleToggleSidebar}
          onToggleThinking={() => {
            if (!isGenerating) {
              setThinkingEnabled((current) => !current);
            }
          }}
          thinkingEnabled={thinkingEnabled}
          title={activeTitle}
        />

        <div className={styles.scrollArea} ref={scrollRef}>
          <div className={styles.content}>
            <ChatView
              activeChatId={activeChatId}
              messages={activeMessages}
              onToggleThinkingPanel={handleToggleThinkingPanel}
              thinkingPanels={thinkingPanels}
            />
          </div>
        </div>

        <Composer
          inputRef={inputRef}
          isGenerating={isGenerating}
          onChange={setInputValue}
          onSend={() => {
            void handleSendMessage();
          }}
          value={inputValue}
        />
      </main>
    </div>
  );
}
