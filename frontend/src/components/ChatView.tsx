"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AttachedImage, ChatMessage } from "@/types/chat";
import { sendChatMessage, checkHealth, ApiError } from "@/lib/api";
import MessageList from "@/components/MessageList";
import MessageComposer from "@/components/MessageComposer";
import WelcomeScreen from "@/components/WelcomeScreen";

function generateId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

interface ChatViewProps {
  /** Conversation to continue, when one is already persisted. */
  conversationId: string | null;
  /** Transcript to show on mount, when the user opened a saved conversation. */
  initialMessages?: ChatMessage[];
  /** Called once, when the first reply creates a conversation for this view. */
  onConversationCreated?: (id: string) => void;
  /**
   * Called after a reply was appended to an already-persisted conversation, so
   * the caller can refresh derived data (message count, updated time).
   */
  onConversationUpdated?: () => void;
  onToggleSidebar: () => void;
}

// ── Backend connectivity status ───────────────────────────────────────────────
type BackendStatus = "checking" | "online" | "offline";

export default function ChatView({
  conversationId,
  initialMessages,
  onConversationCreated,
  onConversationUpdated,
  onToggleSidebar,
}: ChatViewProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages ?? []);
  /**
   * The conversation this view writes to. Seeded from the prop and updated when
   * the backend creates one. Later prop changes are deliberately not followed:
   * the parent remounts this component (via its reset signal) whenever a
   * different conversation should be shown, so following the prop here would
   * clobber the transcript mid-conversation.
   */
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    conversationId
  );
  const [isLoading, setIsLoading] = useState(false);
  const [prefill, setPrefill] = useState<string | undefined>(undefined);
  /** Inline error shown below the message list (network / API failures) */
  const [chatError, setChatError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const bottomRef = useRef<HTMLDivElement>(null);
  /**
   * Whether this ChatView is mounted.
   *
   * A send can resolve after the parent replaced this component (a new
   * selection or New Chat bumps its reset signal), and by then this instance's
   * state updates and `onConversationCreated` callback are stale — the callback
   * would reopen the conversation the user just navigated away from.
   *
   * React Strict Mode runs effects setup → cleanup → setup in development, so
   * the effect re-arms the flag on the second setup.
   */
  const isMountedRef = useRef(false);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // ── Health check on mount ────────────────────────────────────────────────
  useEffect(() => {
    checkHealth().then((healthy) => {
      if (!isMountedRef.current) return;
      setBackendStatus(healthy ? "online" : "offline");
    });
  }, []);

  // ── Scroll to bottom on new messages ────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // ── Send handler ─────────────────────────────────────────────────────────
  const handleSend = useCallback(
    async (text: string, image: AttachedImage | undefined) => {
      // Clear any previous error and prevent duplicate submissions
      setChatError(null);

      const userMsg: ChatMessage = {
        id: generateId(),
        role: "user",
        content: text,
        attachedImage: image,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        // When an image is attached the text may be "(Image attached)"; we still
        // send it alongside the image so the backend gets both. The conversation
        // id is only sent when this view is already part of a saved
        // conversation, so a new chat never appends to someone else's history.
        const data = await sendChatMessage(
          text,
          image?.file,
          activeConversationId ?? undefined
        );

        // The chat area may have been replaced while this request was running
        // (conversation switch or New Chat). Drop the stale reply: neither this
        // transcript nor the parent's active conversation may be touched.
        if (!isMountedRef.current) return;

        const assistantMsg: ChatMessage = {
          id: generateId(),
          role: "assistant",
          content: data.reply,
          sources: data.sources ?? [],
          knowledgeGraph: data.knowledge_graph ?? [],
          timestamp: new Date(),
        };

        setMessages((prev) => [...prev, assistantMsg]);

        // The first persisted reply creates the conversation: remember it so the
        // next message continues the same one, and tell the parent once so it
        // can highlight it and refresh the sidebar.
        if (!activeConversationId && data.conversation_id) {
          setActiveConversationId(data.conversation_id);
          onConversationCreated?.(data.conversation_id);
        } else if (data.conversation_id) {
          // Appended to an existing conversation: its message count and updated
          // time changed, so the list the sidebar renders is now out of date.
          // This never remounts the view — the parent only refetches the list.
          onConversationUpdated?.();
        }
      } catch (err) {
        // Nothing to report to a view that is no longer on screen.
        if (!isMountedRef.current) return;

        // Surface a friendly error inline; keep the user's message visible
        const message =
          err instanceof ApiError
            ? err.message
            : "An unexpected error occurred. Please try again.";
        setChatError(message);
        // Mark backend as offline if it's a network-level failure
        if (err instanceof ApiError && err.status === 0) {
          setBackendStatus("offline");
        }
      } finally {
        if (isMountedRef.current) setIsLoading(false);
      }
    },
    [activeConversationId, onConversationCreated, onConversationUpdated]
  );

  const handleExampleQuestion = (question: string) => {
    setChatError(null);
    setPrefill(question);
  };

  // ── Status badge config ──────────────────────────────────────────────────
  const statusConfig: Record<
    BackendStatus,
    { dot: string; label: string; border: string; bg: string; text: string }
  > = {
    checking: {
      dot: "bg-amber-400 animate-pulse",
      label: "Connecting…",
      border: "border-amber-200",
      bg: "bg-amber-50",
      text: "text-amber-700",
    },
    online: {
      dot: "bg-emerald-500 animate-pulse",
      label: "Backend connected",
      border: "border-emerald-200",
      bg: "bg-emerald-50",
      text: "text-emerald-700",
    },
    offline: {
      dot: "bg-red-400",
      label: "Backend offline",
      border: "border-red-200",
      bg: "bg-red-50",
      text: "text-red-700",
    },
  };
  const status = statusConfig[backendStatus];

  return (
    <div className="flex flex-col flex-1 min-h-0 bg-gray-50">
      {/* ── Top bar ── */}
      <header className="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-3 sm:px-6">
        {/* Mobile hamburger */}
        <button
          id="open-sidebar-btn"
          onClick={onToggleSidebar}
          className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 hover:text-gray-700
                      transition-colors lg:hidden"
          aria-label="Open sidebar"
          aria-controls="sidebar"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <line x1="4" y1="6" x2="20" y2="6" />
            <line x1="4" y1="12" x2="20" y2="12" />
            <line x1="4" y1="18" x2="20" y2="18" />
          </svg>
        </button>

        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-gray-800 truncate">
            {messages.length === 0 ? "New conversation" : "Crop Disease Assistant"}
          </h2>
          <p className="text-[10px] text-gray-400">
            {messages.length === 0
              ? "Ask a question or upload a crop photo"
              : `${messages.length} message${messages.length === 1 ? "" : "s"}`}
          </p>
        </div>

        {/* Backend status badge */}
        <div
          className={`ml-auto flex items-center gap-1.5 rounded-full border px-3 py-1 ${status.border} ${status.bg}`}
          role="status"
          aria-label={`Backend status: ${status.label}`}
        >
          <span className={`h-2 w-2 rounded-full ${status.dot}`} aria-hidden="true" />
          <span className={`text-[10px] font-semibold ${status.text}`}>{status.label}</span>
        </div>
      </header>

      {/* ── Message area ── */}
      <main
        className="flex-1 overflow-y-auto px-4 py-6 sm:px-6"
        aria-label="Chat conversation"
      >
        {messages.length === 0 ? (
          <WelcomeScreen onSelectQuestion={handleExampleQuestion} />
        ) : (
          <div className="mx-auto max-w-2xl">
            <MessageList messages={messages} isLoading={isLoading} />

            {/* ── Inline chat error ── */}
            {chatError && (
              <div
                role="alert"
                className="mt-4 flex items-start gap-3 rounded-xl border border-red-200
                            bg-red-50 px-4 py-3 text-sm text-red-700"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="mt-0.5 shrink-0 text-red-500"
                  aria-hidden="true"
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                <div className="min-w-0 flex-1">
                  <p className="font-semibold">Request failed</p>
                  <p className="mt-0.5 text-xs text-red-600">{chatError}</p>
                </div>
                <button
                  onClick={() => setChatError(null)}
                  className="ml-auto shrink-0 text-red-400 hover:text-red-600 transition-colors"
                  aria-label="Dismiss error"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M18 6 6 18M6 6l12 12" />
                  </svg>
                </button>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </main>

      {/* ── Offline banner (shown above composer when backend is unreachable) ── */}
      {backendStatus === "offline" && (
        <div
          className="border-t border-amber-200 bg-amber-50 px-4 py-2 text-center text-xs text-amber-700"
          role="status"
        >
          ⚠️ Backend not reachable at{" "}
          <code className="font-mono">http://127.0.0.1:8000</code> — make sure the FastAPI
          server is running.
        </div>
      )}

      {/* ── Composer ── */}
      <div className="mx-auto w-full max-w-2xl px-0 sm:px-0">
        <MessageComposer
          onSend={handleSend}
          isLoading={isLoading}
          prefill={prefill}
          onPrefillConsumed={() => setPrefill(undefined)}
        />
      </div>
    </div>
  );
}
