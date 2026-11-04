"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Sidebar from "@/components/Sidebar";
import ChatView from "@/components/ChatView";
import {
  ApiError,
  type ConversationMessage,
  type ConversationSummary,
  deleteConversation,
  getConversation,
  listConversations,
  renameConversation,
} from "@/lib/api";
import { ensureSession } from "@/lib/session";
import { ChatMessage } from "@/types/chat";

/**
 * Convert a stored message into the chat view's model.
 *
 * The backend's `content` is the canonical reply the user saw, so a stored
 * image assessment stays visible inside it, and citations and graph edges are
 * carried over unchanged. Mapping in place preserves the backend's ordering.
 */
function toChatMessage(message: ConversationMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    sources: message.sources?.map((source) => ({
      // `filename` is required by the renderer; the backend already falls back
      // to the title, but a row may carry neither.
      filename: source.filename ?? source.title ?? source.url ?? "Reference",
      citation: source.citation,
      title: source.title,
      url: source.url,
      institution: source.institution,
      evidence_level: source.evidence_level,
      chunk_id: source.chunk_id,
    })),
    knowledgeGraph: message.knowledge_graph,
    storedImage: message.image
      ? {
          id: message.image.id,
          mimeType: message.image.mime_type,
          byteSize: message.image.byte_size,
        }
      : undefined,
    timestamp: new Date(message.created_at),
  };
}

function describeError(error: unknown): string {
  return error instanceof ApiError
    ? error.message
    : "Something went wrong. Please try again.";
}

/**
 * CropSense AI
 *
 * Owns conversation identity and history for the whole page:
 *  - which conversation is open, and the transcript to show for it
 *  - the sidebar list, its loading and error states
 *  - a reset signal used as ChatView's key, so opening a conversation or
 *    starting a new chat remounts the chat area cleanly
 *
 * Messages are sent to the FastAPI backend, which handles RAG retrieval,
 * agricultural knowledge graph queries, and Gemini multimodal vision analysis.
 * Nothing is deleted from the backend unless the user confirms a delete.
 */
export default function Home() {
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    null
  );
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [initialMessages, setInitialMessages] = useState<ChatMessage[]>([]);
  /**
   * Bumped whenever the chat area must be rebuilt from `initialMessages`. Used
   * as ChatView's `key`, so it also clears the pending prefill and inline error.
   */
  const [resetSignal, setResetSignal] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [isLoadingConversations, setIsLoadingConversations] = useState(true);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Id of the most recent conversation-selection request.
   *
   * History requests are not cancelled on the wire, so a slower earlier request
   * can resolve after a newer one. Each selection (and New Chat) takes a new id;
   * a response may only be applied — and only clear the loading indicator — while
   * its id is still the current one.
   */
  const latestSelectionRef = useRef(0);

  // ── Load saved history once, in the browser ─────────────────────────────
  useEffect(() => {
    let cancelled = false;

    (async () => {
      setIsLoadingConversations(true);
      try {
        // Explicit so a backend that is down fails at bootstrap with a clear
        // message; listConversations() would otherwise create the session
        // implicitly on its first request.
        await ensureSession();
        const items = await listConversations();
        if (!cancelled) {
          setConversations(items);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(describeError(err));
      } finally {
        if (!cancelled) setIsLoadingConversations(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const refreshConversations = useCallback(async () => {
    const items = await listConversations();
    setConversations(items);
  }, []);

  // ── Open a saved conversation ───────────────────────────────────────────
  const handleSelectConversation = useCallback(async (id: string) => {
    const requestId = latestSelectionRef.current + 1;
    latestSelectionRef.current = requestId;

    setError(null);
    setIsLoadingConversation(true);
    try {
      const detail = await getConversation(id);

      // A newer selection, or New Chat, started while this was in flight: this
      // result is stale and must not replace what the user opened afterwards.
      if (latestSelectionRef.current !== requestId) return;

      setActiveConversationId(detail.id);
      setInitialMessages(detail.messages.map(toChatMessage));
      // Remount ChatView so it starts from the loaded transcript rather than
      // the previous conversation's messages.
      setResetSignal((value) => value + 1);
    } catch (err) {
      if (latestSelectionRef.current !== requestId) return;
      setError(describeError(err));
    } finally {
      // Only the newest request owns the loading indicator, so an older reply
      // cannot hide the fact that a newer one is still loading.
      if (latestSelectionRef.current === requestId) setIsLoadingConversation(false);
    }
  }, []);

  // ── Start a new chat (deletes nothing) ──────────────────────────────────
  const handleNewChat = useCallback(() => {
    // Invalidate any history request still in flight, and drop its indicator.
    latestSelectionRef.current += 1;
    setIsLoadingConversation(false);

    setActiveConversationId(null);
    setInitialMessages([]);
    setResetSignal((value) => value + 1);
    setError(null);
  }, []);

  // ── The first reply created a conversation ──────────────────────────────
  // ChatView calls this once; it must not bump the reset signal, or the
  // transcript that was just built would be thrown away.
  const handleConversationCreated = useCallback(
    (id: string) => {
      setActiveConversationId(id);
      refreshConversations().catch((err) => setError(describeError(err)));
    },
    [refreshConversations]
  );

  // ── An existing conversation gained a message ───────────────────────────
  // The backend has just changed that row's message_count and updated_at, so the
  // list is refetched. Nothing else is touched: no reset signal, no active id
  // change and no transcript replacement, so ChatView keeps running undisturbed.
  const handleConversationUpdated = useCallback(() => {
    refreshConversations().catch((err) => setError(describeError(err)));
  }, [refreshConversations]);

  // ── Rename ──────────────────────────────────────────────────────────────
  const handleRename = useCallback(async (id: string, title: string) => {
    try {
      const updated = await renameConversation(id, title);
      // Updated in place: a rename does not reorder the list, and this keeps
      // the sidebar scroll position. The error is rethrown so the inline editor
      // can stay open instead of silently discarding the typed title.
      setConversations((prev) =>
        prev.map((conversation) =>
          conversation.id === id
            ? {
                ...conversation,
                title: updated.title,
                updated_at: updated.updated_at,
              }
            : conversation
        )
      );
    } catch (err) {
      setError(describeError(err));
      throw err;
    }
  }, []);

  // ── Delete ──────────────────────────────────────────────────────────────
  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteConversation(id);
        setConversations((prev) =>
          prev.filter((conversation) => conversation.id !== id)
        );
        if (activeConversationId === id) {
          // The open conversation is gone: fall back to an empty new chat.
          setActiveConversationId(null);
          setInitialMessages([]);
          setResetSignal((value) => value + 1);
        }
      } catch (err) {
        setError(describeError(err));
      }
    },
    [activeConversationId]
  );

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar – fixed drawer on mobile, static column on desktop */}
      <Sidebar
        conversations={conversations}
        activeConvId={activeConversationId}
        onSelect={handleSelectConversation}
        onDelete={handleDelete}
        onRename={handleRename}
        onNewChat={handleNewChat}
        isLoading={isLoadingConversations}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main chat area */}
      <div className="flex flex-1 flex-col min-w-0 lg:ml-0">
        {/* History errors: the chat itself stays usable */}
        {error && (
          <div
            role="alert"
            className="flex items-start gap-2 border-b border-red-200 bg-red-50 px-4 py-2 sm:px-6"
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
              className="mt-0.5 shrink-0 text-red-500"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <p className="min-w-0 flex-1 text-xs text-red-700">{error}</p>
            <button
              onClick={() => setError(null)}
              className="shrink-0 text-red-400 transition-colors hover:text-red-600"
              aria-label="Dismiss history error"
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

        {isLoadingConversation && (
          <div
            role="status"
            className="border-b border-emerald-200 bg-emerald-50 px-4 py-1.5 text-center
                        text-[11px] text-emerald-700 sm:px-6"
          >
            Loading conversation…
          </div>
        )}

        <ChatView
          key={resetSignal}
          conversationId={activeConversationId}
          initialMessages={initialMessages}
          onConversationCreated={handleConversationCreated}
          onConversationUpdated={handleConversationUpdated}
          onToggleSidebar={() => setSidebarOpen(true)}
        />
      </div>
    </div>
  );
}
