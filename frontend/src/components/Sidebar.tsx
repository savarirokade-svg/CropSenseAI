"use client";

import Image from "next/image";
import {
  useEffect,
  useState,
  useSyncExternalStore,
  type FormEvent,
} from "react";
import { ConversationSummary } from "@/lib/api";

interface SidebarProps {
  /** Saved conversations belonging to this browser's session, newest first. */
  conversations: ConversationSummary[];
  /** Currently open conversation, or null for an unsaved new chat. */
  activeConvId: string | null;
  onSelect: (id: string) => void;
  /** Resolves once the deletion has been handled; the dialog always closes. */
  onDelete: (id: string) => Promise<void>;
  /** Rejects when the rename failed, so the editor can stay open. */
  onRename: (id: string, title: string) => Promise<void>;
  onNewChat: () => void;
  /** True while the conversation list is loading. */
  isLoading: boolean;
  /** Whether the mobile drawer is open */
  isOpen: boolean;
  onClose: () => void;
}

function formatRelativeDate(isoString: string, now: number): string {
  const time = new Date(isoString).getTime();
  if (Number.isNaN(time)) return "";
  const diff = now - time;
  const days = Math.floor(diff / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  return new Date(isoString).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

// Client-side clock for the relative "Today / Yesterday / n days ago" labels.
// Reading Date.now() during render would make the prerendered HTML disagree with
// the first client render, so it is exposed as an external store instead: the
// server snapshot is 0 (labels render empty) and the browser value is read after
// hydration. Cached so the snapshot is stable across renders.
const NO_SUBSCRIPTION = () => () => {};
let clientNow = 0;

function getBrowserNow(): number {
  if (clientNow === 0) {
    clientNow = Date.now();
  }
  return clientNow;
}

function getServerNow(): number {
  return 0;
}

export default function Sidebar({
  conversations,
  activeConvId,
  onSelect,
  onDelete,
  onRename,
  onNewChat,
  isLoading,
  isOpen,
  onClose,
}: SidebarProps) {
  const now = useSyncExternalStore(NO_SUBSCRIPTION, getBrowserNow, getServerNow);

  /** Conversation whose title is being edited inline, if any. */
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  /** Conversation awaiting delete confirmation, if any. */
  const [pendingDelete, setPendingDelete] =
    useState<ConversationSummary | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Escape closes the confirmation dialog without deleting anything.
  useEffect(() => {
    if (!pendingDelete) return;
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPendingDelete(null);
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [pendingDelete]);

  const startRename = (conversation: ConversationSummary) => {
    setEditingId(conversation.id);
    setDraftTitle(conversation.title);
  };

  const cancelRename = () => {
    setEditingId(null);
    setDraftTitle("");
  };

  const submitRename = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!editingId || isSaving) return;

    const title = draftTitle.trim();
    // Mirrors the backend's 1–200 rule so most mistakes never leave the page.
    if (title.length < 1 || title.length > 200) return;

    setIsSaving(true);
    try {
      await onRename(editingId, title);
      cancelRename();
    } catch {
      // The page shows the reason; keep the editor open so it can be fixed.
    } finally {
      setIsSaving(false);
    }
  };

  const confirmDelete = async () => {
    if (!pendingDelete || isDeleting) return;
    setIsDeleting(true);
    try {
      await onDelete(pendingDelete.id);
    } finally {
      setIsDeleting(false);
      setPendingDelete(null);
    }
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/50 lg:hidden"
          aria-hidden="true"
          onClick={onClose}
        />
      )}

      {/* Sidebar panel */}
      <aside
        id="sidebar"
        aria-label="Sidebar navigation"
        className={[
          "fixed inset-y-0 left-0 z-30 flex w-72 flex-col bg-[#0f2318] text-white",
          "transform transition-transform duration-300 ease-in-out",
          "lg:static lg:translate-x-0 lg:transition-none",
          isOpen ? "translate-x-0" : "-translate-x-full",
        ].join(" ")}
      >
        {/* ── Logo & brand ── */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-white/10">
          <div className="relative h-9 w-9 shrink-0 overflow-hidden rounded-lg">
            <Image
              src="/logo.jpg"
              alt="CropSense AI logo"
              fill
              sizes="36px"
              className="object-cover"
              priority
            />
          </div>
          <div className="min-w-0">
            <span className="block text-sm font-bold tracking-wide text-emerald-300">
              CropSense AI
            </span>
            <span className="block text-[10px] text-white/40 uppercase tracking-wider">
              Crop Disease Assistant
            </span>
          </div>
          {/* Mobile close button */}
          <button
            className="ml-auto -mr-1 rounded-md p-1.5 text-white/50 hover:text-white hover:bg-white/10 transition-colors lg:hidden"
            onClick={onClose}
            aria-label="Close sidebar"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
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

        {/* ── New Chat button ── */}
        <div className="px-4 pt-4 pb-2">
          <button
            id="new-chat-btn"
            onClick={() => {
              onNewChat();
              onClose();
            }}
            className={[
              "w-full flex items-center justify-center gap-2 rounded-xl px-4 py-2.5",
              "bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700",
              "text-sm font-semibold text-white",
              "transition-colors duration-150 focus-visible:outline focus-visible:outline-2",
              "focus-visible:outline-offset-2 focus-visible:outline-emerald-400",
            ].join(" ")}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M12 5v14M5 12h14" />
            </svg>
            New Chat
          </button>
        </div>

        {/* ── Conversation list ── */}
        <nav aria-label="Conversation history" className="flex-1 overflow-y-auto px-2 py-2">
          <p className="px-3 py-1 text-[10px] font-semibold uppercase tracking-widest text-white/30">
            History
          </p>

          {isLoading ? (
            <ul role="list" aria-hidden="true" className="mt-1 space-y-1 px-1">
              {[0, 1, 2, 3].map((row) => (
                <li key={row} className="rounded-lg px-2 py-2.5">
                  <div className="h-3 w-3/4 animate-pulse rounded bg-white/10" />
                  <div className="mt-2 h-2 w-1/2 animate-pulse rounded bg-white/5" />
                </li>
              ))}
            </ul>
          ) : conversations.length === 0 ? (
            <div className="px-3 py-6 text-center">
              <p className="text-sm text-white/60">No saved conversations yet</p>
              <p className="mt-1 text-[10px] leading-relaxed text-white/30">
                Ask a question in a new chat and it will appear here.
              </p>
            </div>
          ) : (
            <ul role="list" className="mt-1 space-y-0.5">
              {conversations.map((conv) => (
                <li key={conv.id} className="group relative">
                  {editingId === conv.id ? (
                    <form
                      onSubmit={submitRename}
                      className="rounded-lg bg-white/5 px-2 py-2"
                    >
                      <label
                        htmlFor={`rename-${conv.id}`}
                        className="sr-only"
                      >
                        Conversation title
                      </label>
                      <input
                        id={`rename-${conv.id}`}
                        autoFocus
                        value={draftTitle}
                        maxLength={200}
                        onChange={(event) => setDraftTitle(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") {
                            event.preventDefault();
                            cancelRename();
                          }
                        }}
                        className="w-full rounded-md border border-emerald-400/60 bg-[#0f2318]
                                    px-2 py-1 text-sm text-white outline-none
                                    focus:border-emerald-400 focus:ring-2 focus:ring-emerald-400/30"
                      />
                      <div className="mt-2 flex items-center justify-end gap-2">
                        <button
                          type="button"
                          onClick={cancelRename}
                          className="rounded-md px-2 py-1 text-[11px] text-white/60
                                      hover:bg-white/10 hover:text-white
                                      focus-visible:outline focus-visible:outline-2
                                      focus-visible:outline-emerald-400"
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          disabled={isSaving}
                          className="rounded-md bg-emerald-600 px-2.5 py-1 text-[11px]
                                      font-semibold text-white hover:bg-emerald-500
                                      disabled:cursor-not-allowed disabled:opacity-60
                                      focus-visible:outline focus-visible:outline-2
                                      focus-visible:outline-emerald-400"
                        >
                          {isSaving ? "Saving…" : "Save"}
                        </button>
                      </div>
                    </form>
                  ) : (
                    <>
                      <button
                        id={`conv-${conv.id}`}
                        onClick={() => {
                          onSelect(conv.id);
                          onClose();
                        }}
                        aria-current={activeConvId === conv.id ? "page" : undefined}
                        className={[
                          "w-full text-left rounded-lg px-3 py-2.5 transition-colors duration-100",
                          "group/item flex flex-col gap-1",
                          "focus-visible:outline focus-visible:outline-2",
                          "focus-visible:outline-offset-2 focus-visible:outline-emerald-400",
                          activeConvId === conv.id
                            ? "bg-emerald-700/50 text-white"
                            : "text-white/70 hover:bg-white/8 hover:text-white",
                        ].join(" ")}
                      >
                        <span className="block truncate pr-14 text-sm leading-snug">
                          {conv.title}
                        </span>
                        <span className="flex flex-wrap items-center gap-x-1.5 gap-y-1 pr-8 text-[10px] text-white/35 group-hover/item:text-white/50">
                          <span>{now > 0 ? formatRelativeDate(conv.updated_at, now) : ""}</span>
                          <span aria-hidden="true">·</span>
                          <span>
                            {conv.message_count} message
                            {conv.message_count === 1 ? "" : "s"}
                          </span>
                          {/* has_diagnosis only means an image was analysed — it is
                              not a confirmed disease diagnosis. */}
                          {conv.has_diagnosis && (
                            <span
                              className="rounded-full border border-emerald-400/30 bg-emerald-400/10
                                          px-1.5 py-0.5 text-[9px] font-medium text-emerald-300"
                              title="This conversation includes an image analysis"
                            >
                              Image analyzed
                            </span>
                          )}
                        </span>
                      </button>

                      {/* Per-conversation actions. Always rendered (not hover-only) so
                          they stay reachable by keyboard and on touch screens. */}
                      <div className="absolute right-1 top-1 flex items-center gap-0.5 opacity-70 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                        <button
                          type="button"
                          onClick={() => startRename(conv)}
                          aria-label={`Rename ${conv.title}`}
                          title="Rename"
                          className="rounded-md p-1.5 text-white/50 hover:bg-white/10 hover:text-white
                                      transition-colors focus-visible:outline focus-visible:outline-2
                                      focus-visible:outline-emerald-400"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="13"
                            height="13"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            aria-hidden="true"
                          >
                            <path d="M12 20h9" />
                            <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
                          </svg>
                        </button>
                        <button
                          type="button"
                          onClick={() => setPendingDelete(conv)}
                          aria-label={`Delete ${conv.title}`}
                          title="Delete"
                          className="rounded-md p-1.5 text-white/50 hover:bg-red-500/20 hover:text-red-300
                                      transition-colors focus-visible:outline focus-visible:outline-2
                                      focus-visible:outline-red-400"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="13"
                            height="13"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            aria-hidden="true"
                          >
                            <path d="M3 6h18" />
                            <path d="M8 6V4h8v2" />
                            <path d="M19 6l-1 14H6L5 6" />
                          </svg>
                        </button>
                      </div>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
        </nav>

        {/* ── Footer disclaimer ── */}
        <div className="border-t border-white/10 px-4 py-3">
          <p className="text-[10px] leading-relaxed text-white/30 text-center">
            AI Assistant · Educational guidance.
            <br />
            Not for certified crop diagnosis.
          </p>
        </div>
      </aside>

      {/* ── Delete confirmation ──
          Rendered outside the panel: the aside is a transformed element, so a
          `fixed` child would be positioned relative to it instead of the screen. */}
      {pendingDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4"
          onClick={(event) => {
            if (event.target === event.currentTarget) setPendingDelete(null);
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-conversation-title"
            aria-describedby="delete-conversation-description"
            className="w-full max-w-sm rounded-2xl bg-white p-5 shadow-xl"
          >
            <h2
              id="delete-conversation-title"
              className="text-sm font-semibold text-gray-900"
            >
              Delete this conversation?
            </h2>
            <p
              id="delete-conversation-description"
              className="mt-2 text-xs leading-relaxed text-gray-600"
            >
              “{pendingDelete.title}” and its messages will be permanently
              deleted. This cannot be undone.
            </p>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                autoFocus
                onClick={() => setPendingDelete(null)}
                disabled={isDeleting}
                className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium
                            text-gray-700 hover:bg-gray-50 disabled:opacity-60
                            focus-visible:outline focus-visible:outline-2
                            focus-visible:outline-offset-2 focus-visible:outline-gray-400"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmDelete}
                disabled={isDeleting}
                className="rounded-lg bg-red-600 px-3 py-1.5 text-xs font-semibold text-white
                            hover:bg-red-500 disabled:cursor-not-allowed disabled:opacity-60
                            focus-visible:outline focus-visible:outline-2
                            focus-visible:outline-offset-2 focus-visible:outline-red-500"
              >
                {isDeleting ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
