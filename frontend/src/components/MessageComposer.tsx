"use client";

import {
  useRef,
  useState,
  useCallback,
  useEffect,
  type ChangeEvent,
  type KeyboardEvent,
  type FormEvent,
} from "react";
import { AttachedImage } from "@/types/chat";

interface MessageComposerProps {
  onSend: (text: string, image: AttachedImage | undefined) => void;
  isLoading: boolean;
  /** Pre-fill the input (used by example questions) */
  prefill?: string;
  onPrefillConsumed: () => void;
}

export default function MessageComposer({
  onSend,
  isLoading,
  prefill,
  onPrefillConsumed,
}: MessageComposerProps) {
  const [text, setText] = useState("");
  const [attachedImage, setAttachedImage] = useState<AttachedImage | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // ── Apply prefill from example questions ──────────────────────────────────
  // "Adjust state during render" (the documented React pattern for deriving
  // state from a changing prop): React re-runs this component immediately with
  // the new value and never commits the intermediate result, so the textarea
  // shows the prefill in the same paint and no effect has to write state.
  const [appliedPrefill, setAppliedPrefill] = useState<string | undefined>(undefined);
  if (prefill !== appliedPrefill) {
    setAppliedPrefill(prefill);
    if (prefill) {
      setText(prefill);
    }
  }

  // Focus and acknowledge the prefill after the value has been committed.
  // The parent clears `prefill`, so clicking the same example twice re-applies it.
  useEffect(() => {
    if (!prefill) return;
    textareaRef.current?.focus();
    onPrefillConsumed();
  }, [prefill, onPrefillConsumed]);

  // ── Auto-resize textarea ──────────────────────────────────────────────────
  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    setError(null);
    autoResize();
  };

  // ── Send on Enter (Shift+Enter for newline) ───────────────────────────────
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // ── Image attachment ──────────────────────────────────────────────────────
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate: must be an image and ≤ 10 MB
    if (!file.type.startsWith("image/")) {
      setError("Only image files are supported.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("Image must be smaller than 10 MB.");
      return;
    }

    setError(null);
    const previewUrl = URL.createObjectURL(file);
    setAttachedImage({ file, previewUrl, fileName: file.name });
    // Reset input so the same file can be reselected
    e.target.value = "";
  };

  const removeAttachment = () => {
    if (attachedImage) {
      URL.revokeObjectURL(attachedImage.previewUrl);
    }
    setAttachedImage(undefined);
  };

  // ── Submit ─────────────────────────────────────────────────────────────────
  const handleSubmit = (e?: FormEvent) => {
    e?.preventDefault();
    const trimmed = text.trim();
    if (!trimmed && !attachedImage) {
      setError("Please enter a message or attach an image.");
      return;
    }
    if (isLoading) return;

    onSend(trimmed || "(Image attached)", attachedImage);
    setText("");
    setAttachedImage(undefined);
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const canSend = (text.trim().length > 0 || !!attachedImage) && !isLoading;

  return (
    <div className="border-t border-gray-200 bg-white px-4 pt-3 pb-4 sm:px-6">
      {/* ── Image preview ── */}
      {attachedImage && (
        <div className="mb-3 flex items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={attachedImage.previewUrl}
            alt={`Preview: ${attachedImage.fileName}`}
            className="h-12 w-12 rounded-lg object-cover border border-emerald-200"
          />
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-medium text-emerald-800">
              📎 {attachedImage.fileName}
            </p>
            <p className="text-[10px] text-emerald-600">Ready to send</p>
          </div>
          <button
            type="button"
            onClick={removeAttachment}
            aria-label="Remove attached image"
            className="rounded-full p-1 text-emerald-600 hover:bg-emerald-100 hover:text-red-500 transition-colors"
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
              aria-hidden="true"
            >
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      {/* ── Error message ── */}
      {error && (
        <div
          role="alert"
          className="mb-2 flex items-center gap-2 rounded-lg bg-red-50 border border-red-200
                      px-3 py-2 text-xs text-red-700"
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
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-auto text-red-500 hover:text-red-700"
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}

      {/* ── Composer form ── */}
      <form
        onSubmit={handleSubmit}
        className="flex items-end gap-2 rounded-2xl border border-gray-200 bg-gray-50
                    p-2 shadow-sm focus-within:border-emerald-400 focus-within:ring-2
                    focus-within:ring-emerald-100 transition-all duration-150"
      >
        {/* Attachment button */}
        <button
          type="button"
          id="attach-image-btn"
          onClick={() => fileInputRef.current?.click()}
          aria-label="Attach crop image"
          disabled={isLoading}
          className={[
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl",
            "text-gray-400 transition-colors duration-150",
            isLoading
              ? "cursor-not-allowed opacity-40"
              : "hover:bg-emerald-100 hover:text-emerald-600",
          ].join(" ")}
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
            <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
            <circle cx="9" cy="9" r="2" />
            <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
          </svg>
        </button>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          aria-hidden="true"
          tabIndex={-1}
          onChange={handleFileChange}
        />

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          id="message-input"
          value={text}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          placeholder="Ask about a crop disease, symptoms, or treatment…"
          aria-label="Message input"
          rows={1}
          disabled={isLoading}
          className={[
            "flex-1 resize-none bg-transparent text-sm text-gray-800 placeholder-gray-400",
            "outline-none leading-relaxed py-2 px-1 max-h-[200px] min-h-[36px]",
            "disabled:cursor-not-allowed",
          ].join(" ")}
        />

        {/* Send button */}
        <button
          type="submit"
          id="send-message-btn"
          disabled={!canSend}
          aria-label="Send message"
          className={[
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl",
            "transition-all duration-150",
            canSend
              ? "bg-emerald-600 text-white hover:bg-emerald-500 shadow-sm hover:shadow-emerald-300/50"
              : "bg-gray-200 text-gray-400 cursor-not-allowed",
          ].join(" ")}
        >
          {isLoading ? (
            <svg
              className="animate-spin"
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
          ) : (
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
              aria-hidden="true"
            >
              <path d="m22 2-7 20-4-9-9-4Z" />
              <path d="M22 2 11 13" />
            </svg>
          )}
        </button>
      </form>

      {/* Keyboard hint */}
      <p className="mt-2 text-center text-[10px] text-gray-400">
        Press{" "}
        <kbd className="rounded border border-gray-200 bg-gray-100 px-1 py-0.5 font-mono text-[9px]">
          Enter
        </kbd>{" "}
        to send ·{" "}
        <kbd className="rounded border border-gray-200 bg-gray-100 px-1 py-0.5 font-mono text-[9px]">
          Shift+Enter
        </kbd>{" "}
        for new line
      </p>
    </div>
  );
}
