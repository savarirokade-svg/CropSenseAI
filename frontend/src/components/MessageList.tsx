"use client";

import { useEffect, useState } from "react";
import { ChatMessage, ChatSource, StoredImage } from "@/types/chat";
import { fetchImage } from "@/lib/api";

// ---------------------------------------------------------------------------
// Minimal Markdown renderer (bold, italic, code, headings, lists, tables)
// Uses no external deps so the bundle stays tiny in Phase 1.
// ---------------------------------------------------------------------------

function renderMarkdown(text: string): React.ReactNode[] {
  const lines = text.split("\n");
  const nodes: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Heading
    const headingMatch = line.match(/^(#{1,3})\s+(.+)/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const content = headingMatch[2];
      const Tag = (`h${level}`) as "h1" | "h2" | "h3";
      const cls =
        level === 1
          ? "text-lg font-bold text-gray-900 mt-3 mb-1"
          : level === 2
            ? "text-base font-bold text-gray-800 mt-3 mb-1"
            : "text-sm font-semibold text-gray-700 mt-2 mb-0.5";
      nodes.push(
        <Tag key={i} className={cls}>
          {inlineMarkdown(content)}
        </Tag>
      );
      i++;
      continue;
    }

    // Table
    if (line.startsWith("|") && i + 1 < lines.length && lines[i + 1].startsWith("|---")) {
      const headerCells = parseCells(line);
      const rows: string[][] = [];
      i += 2; // skip separator
      while (i < lines.length && lines[i].startsWith("|")) {
        rows.push(parseCells(lines[i]));
        i++;
      }
      nodes.push(
        <div key={`table-${i}`} className="my-3 overflow-x-auto rounded-lg border border-gray-200">
          <table className="w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                {headerCells.map((c, ci) => (
                  <th
                    key={ci}
                    className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200"
                  >
                    {inlineMarkdown(c)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr key={ri} className={ri % 2 === 1 ? "bg-gray-50" : ""}>
                  {row.map((c, ci) => (
                    <td key={ci} className="px-3 py-2 text-gray-700 border-b border-gray-100">
                      {inlineMarkdown(c)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }

    // Blockquote
    if (line.startsWith(">")) {
      const content = line.replace(/^>\s*/, "");
      nodes.push(
        <blockquote
          key={i}
          className="my-2 border-l-4 border-amber-400 bg-amber-50 px-4 py-2 text-xs text-amber-800 rounded-r-lg"
        >
          {inlineMarkdown(content)}
        </blockquote>
      );
      i++;
      continue;
    }

    // Unordered list
    if (/^[-*]\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*]\s/, ""));
        i++;
      }
      nodes.push(
        <ul key={`ul-${i}`} className="my-2 space-y-1 pl-5 list-disc text-sm text-gray-700">
          {items.map((item, ii) => (
            <li key={ii}>{inlineMarkdown(item)}</li>
          ))}
        </ul>
      );
      continue;
    }

    // Ordered list
    if (/^\d+\.\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s/, ""));
        i++;
      }
      nodes.push(
        <ol key={`ol-${i}`} className="my-2 space-y-1 pl-5 list-decimal text-sm text-gray-700">
          {items.map((item, ii) => (
            <li key={ii}>{inlineMarkdown(item)}</li>
          ))}
        </ol>
      );
      continue;
    }

    // Empty line → paragraph break
    if (line.trim() === "") {
      nodes.push(<div key={i} className="h-2" />);
      i++;
      continue;
    }

    // Paragraph
    nodes.push(
      <p key={i} className="text-sm text-gray-700 leading-relaxed">
        {inlineMarkdown(line)}
      </p>
    );
    i++;
  }

  return nodes;
}

function parseCells(row: string): string[] {
  return row
    .split("|")
    .slice(1, -1)
    .map((c) => c.trim());
}

function inlineMarkdown(text: string): React.ReactNode {
  // Bold **text** and *italic* and `code`
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-gray-900">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code
          key={i}
          className="rounded bg-gray-100 px-1 py-0.5 font-mono text-xs text-emerald-700"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    return part;
  });
}

// ---------------------------------------------------------------------------
// Source list
// ---------------------------------------------------------------------------

const EVIDENCE_LABELS: Record<string, string> = {
  full_page: "Full page",
  abstract_only: "Abstract only",
};

/**
 * Only http(s) URLs are rendered as links, so a missing, relative or malformed
 * value can never produce a broken or unsafe `href`.
 */
function safeExternalUrl(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
    return trimmed;
  } catch {
    return null;
  }
}

/** Deduplicate by the most specific identity the backend provides. */
function sourceIdentity(source: ChatSource): string | null {
  if (typeof source.chunk_id === "string" && source.chunk_id.trim()) {
    return `chunk:${source.chunk_id.trim()}`;
  }
  if (typeof source.citation === "number" && Number.isInteger(source.citation)) {
    return `citation:${source.citation}`;
  }
  const url = safeExternalUrl(source.url);
  if (url) return `url:${url}`;
  if (typeof source.filename === "string" && source.filename.trim()) {
    return `filename:${source.filename.trim()}`;
  }
  return null;
}

function uniqueSources(sources: ChatSource[]): ChatSource[] {
  const seen = new Set<string>();
  const unique: ChatSource[] = [];
  sources.forEach((source, index) => {
    const key = sourceIdentity(source) ?? `index:${index}`;
    if (seen.has(key)) return;
    seen.add(key);
    unique.push(source);
  });
  return unique;
}

function SourceItem({ source }: { source: ChatSource }) {
  const citation =
    typeof source.citation === "number" &&
    Number.isInteger(source.citation) &&
    source.citation > 0
      ? source.citation
      : null;

  const rawTitle = typeof source.title === "string" ? source.title.trim() : "";
  const rawFilename =
    typeof source.filename === "string" ? source.filename.trim() : "";
  const title = rawTitle || rawFilename || "Untitled source";

  const institution =
    typeof source.institution === "string" ? source.institution.trim() : "";

  const rawEvidence =
    typeof source.evidence_level === "string" ? source.evidence_level.trim() : "";
  const evidenceLabel = rawEvidence
    ? EVIDENCE_LABELS[rawEvidence] ?? rawEvidence.replace(/_/g, " ")
    : "";

  const url = safeExternalUrl(source.url);

  return (
    <li className="rounded-lg border border-gray-100 bg-gray-50 px-3 py-2">
      <div className="flex items-start gap-2">
        <span className="mt-0.5 shrink-0 rounded bg-emerald-100 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-emerald-700">
          {citation !== null ? `Source [${citation}]` : "Source"}
        </span>
        <span className="min-w-0 flex-1 text-xs font-medium leading-snug text-gray-800">
          {title}
        </span>
      </div>

      {institution && (
        <p className="mt-1 text-[10px] leading-relaxed text-gray-500">
          {institution}
        </p>
      )}

      {(evidenceLabel || url) && (
        <div className="mt-1.5 flex flex-wrap items-center gap-2">
          {evidenceLabel && (
            <span className="rounded-full border border-gray-200 bg-white px-2 py-0.5 text-[10px] text-gray-500">
              {evidenceLabel}
            </span>
          )}
          {url && (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[10px] font-medium text-emerald-700 underline decoration-emerald-300
                         underline-offset-2 hover:text-emerald-800"
            >
              View source ↗
            </a>
          )}
        </div>
      )}
    </li>
  );
}

// ---------------------------------------------------------------------------
// Stored image (history)
// ---------------------------------------------------------------------------

/**
 * Renders an image that is stored in the backend.
 *
 * The bytes are fetched through the API client so the session token travels in
 * the request header, then shown from a temporary object URL that is revoked
 * when the bubble unmounts or the image changes. The token never appears in a
 * URL, an attribute or the DOM.
 */
function StoredImagePreview({ image }: { image: StoredImage }) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let createdUrl: string | null = null;
    let cancelled = false;

    fetchImage(image.id)
      .then((blob) => {
        if (cancelled) return;
        createdUrl = URL.createObjectURL(blob);
        setObjectUrl(createdUrl);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });

    return () => {
      cancelled = true;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [image.id]);

  if (!objectUrl) {
    return (
      <div
        className="mb-2 flex h-24 items-center justify-center rounded-xl border border-white/20
                    bg-black/10 text-[10px] text-white/70"
        role="img"
        aria-label={failed ? "Saved image unavailable" : "Loading saved image"}
      >
        {failed ? "Image unavailable" : "Loading image…"}
      </div>
    );
  }

  return (
    <div className="mb-2 overflow-hidden rounded-xl border border-white/20">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={objectUrl}
        alt="Attached crop image"
        className="max-h-48 w-full object-cover"
      />
      <p className="bg-black/20 px-2 py-1 text-[10px] text-white/80">
        📎 Saved image · {Math.max(1, Math.round(image.byteSize / 1024))} KB
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Individual message bubble
// ---------------------------------------------------------------------------

interface MessageBubbleProps {
  message: ChatMessage;
}

function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div
      className={[
        "flex w-full gap-3",
        isUser ? "flex-row-reverse" : "flex-row",
      ].join(" ")}
    >
      {/* Avatar */}
      <div
        className={[
          "mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold",
          isUser
            ? "bg-emerald-600 text-white"
            : "bg-gradient-to-br from-green-700 to-emerald-500 text-white shadow-sm",
        ].join(" ")}
        aria-hidden="true"
      >
        {isUser ? "U" : "🌿"}
      </div>

      {/* Bubble */}
      <div
        className={[
          "max-w-[80%] rounded-2xl px-4 py-3 shadow-sm",
          isUser
            ? "bg-emerald-600 text-white rounded-tr-sm"
            : "bg-white border border-gray-200 rounded-tl-sm",
        ].join(" ")}
      >
        {/* Attached image (user side) */}
        {message.attachedImage && (
          <div className="mb-2 overflow-hidden rounded-xl border border-white/20">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={message.attachedImage.previewUrl}
              alt={`Attached: ${message.attachedImage.fileName}`}
              className="max-h-48 w-full object-cover"
            />
            <p className="bg-black/20 px-2 py-1 text-[10px] text-white/80">
              📎 {message.attachedImage.fileName}
            </p>
          </div>
        )}

        {/* Stored image (history) */}
        {message.storedImage && (
          <StoredImagePreview image={message.storedImage} />
        )}


        {/* Content */}
        {isUser ? (
          <p className="text-sm leading-relaxed text-white whitespace-pre-wrap">
            {message.content}
          </p>
        ) : (
          <>
            <div className="prose-sm">
              {renderMarkdown(message.content)}
            </div>

            {/* Sources */}


            {/* Knowledge Graph */}
            {message.knowledgeGraph && message.knowledgeGraph.length > 0 && (
              <div className="mt-4 border-t border-gray-100 pt-4">
                <div className="mb-3 flex items-center gap-2">
                  <span className="text-lg">🕸️</span>
                  <p className="text-sm font-semibold text-gray-700">
                    Knowledge Graph
                  </p>
                  <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-500">
                    {message.knowledgeGraph.length} relationships
                  </span>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {message.knowledgeGraph.map((item, index) => (
                    <div
                      key={`${item.subject}-${item.relation}-${item.object}-${index}`}
                      className="rounded-xl border border-gray-200 bg-white p-3 shadow-sm"
                    >
                      {/* Subject node */}
                      <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-center">
                        <p className="text-xs text-emerald-700">Subject</p>
                        <p className="text-sm font-semibold text-emerald-900">
                          {item.subject}
                        </p>
                      </div>

                      {/* Relationship */}
                      <div className="flex flex-col items-center py-2">
                        <span className="max-w-full rounded-full bg-gray-100 px-3 py-1 text-center text-xs font-medium text-gray-600">
                          {item.relation.replace(/_/g, " ")}
                        </span>
                        <span className="text-lg leading-5 text-gray-400">
                          ↓
                        </span>
                      </div>

                      {/* Object node */}
                      <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-center">
                        <p className="text-xs text-blue-700">Object</p>
                        <p className="text-sm font-semibold text-blue-900">
                          {item.object}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}




            {message.sources && message.sources.length > 0 && (
              <div className="mt-3 border-t border-gray-100 pt-3">
                <p className="mb-2 text-xs font-semibold text-gray-600">
                  📚 Sources
                </p>

                <ul className="space-y-1.5">
                  {uniqueSources(message.sources).map((source, index) => (
                    <SourceItem
                      key={sourceIdentity(source) ?? `source-${index}`}
                      source={source}
                    />
                  ))}
                </ul>

                <p className="mt-2 text-[10px] text-gray-400">
                  Hybrid retrieval using semantic and keyword search.
                </p>
              </div>
            )}
          </>
        )}


        {/* Timestamp */}
        <p
          className={[
            "mt-1.5 text-[10px]",
            isUser ? "text-white/60 text-right" : "text-gray-400",
          ].join(" ")}
        >
          {message.timestamp.toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------

export function TypingIndicator() {
  return (
    <div className="flex w-full gap-3" role="status" aria-label="Assistant is typing">
      <div
        className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full
                    bg-gradient-to-br from-green-700 to-emerald-500 text-white shadow-sm"
        aria-hidden="true"
      >
        🌿
      </div>
      <div className="max-w-[80%] rounded-2xl rounded-tl-sm bg-white border border-gray-200 px-5 py-4 shadow-sm">
        <div className="flex items-center gap-1" aria-hidden="true">
          {[0, 150, 300].map((delay) => (
            <span
              key={delay}
              className="block h-2 w-2 rounded-full bg-emerald-500 opacity-70"
              style={{
                animation: `bounce 1s ${delay}ms ease-in-out infinite`,
              }}
            />
          ))}
        </div>
        <span className="sr-only">CropSense AI is typing…</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Message list (exported)
// ---------------------------------------------------------------------------

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
}

export default function MessageList({ messages, isLoading }: MessageListProps) {
  return (
    <div
      className="space-y-4"
      role="log"
      aria-label="Chat messages"
      aria-live="polite"
    >
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      {isLoading && <TypingIndicator />}
    </div>
  );
}
