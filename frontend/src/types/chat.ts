
export type MessageRole = "user" | "assistant";

/**
 * An image the user attached to a message.
 * `file` is the real File object needed to upload to the backend.
 * `previewUrl` is a local object-URL used only for display (revoked after use).
 */
export interface AttachedImage {
  file: File;
  previewUrl: string;
  fileName: string;
}

/**
 * An image already stored by the backend, shown when a conversation is loaded
 * from history.
 *
 * The bytes are fetched with the session token and rendered from a temporary
 * object URL: a plain `<img src="/images/{id}">` cannot send the
 * `X-Session-Token` header that the image endpoint requires.
 */
export interface StoredImage {
  id: string;
  mimeType: string;
  byteSize: number;
}

/** A source document returned by POST /chat */
export interface ChatSource {
  filename: string;
  /** Citation number used in the reply, e.g. `[1]` for `citation: 1`. */
  citation?: number;
  /** Full document title; `filename` already falls back to this. */
  title?: string | null;
  /** External document URL. May be missing or unusable; validate before linking. */
  url?: string | null;
  /** Publishing institution, when the source record carries one. */
  institution?: string | null;
  /** Retrieval evidence level, e.g. `"full_page"` or `"abstract_only"`. */
  evidence_level?: string | null;
  /** Stable knowledge-base chunk identifier. */
  chunk_id?: string | null;
  /**
   * Omitted by the backend: the retrieval pipeline has no trustworthy
   * confidence value to report, so the key is absent from /chat responses.
   */
  match_score?: number;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  /** Optional image the user attached when sending this message */
  attachedImage?: AttachedImage;
  /** Image loaded from saved history for this message */
  storedImage?: StoredImage;
  /** Local agricultural documents used to answer this message */
  sources?: ChatSource[];
  /** Knowledge Graph relationships returned by the backend */
  knowledgeGraph?: {
    subject: string;
    relation: string;
    object: string;
  }[];
  timestamp: Date;
}

/** A sidebar conversation entry (used by Sidebar and mockData) */
export interface Conversation {
  id: string;
  title: string;
  updatedAt: string; // ISO 8601
}
