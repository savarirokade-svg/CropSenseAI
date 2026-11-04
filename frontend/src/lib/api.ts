// ---------------------------------------------------------------------------
// CropSense AI — API client
//
// Wraps all HTTP calls to the FastAPI backend so that ChatView.tsx never
// contains raw fetch logic.
//
// Every endpoint below except GET /health needs an anonymous session: the token
// is created on first use (see ./session), sent as X-Session-Token, and replaced
// once automatically if the server rejects it. Tokens are only ever put in the
// request header — they never appear in error messages, in thrown errors or in
// console output.
// ---------------------------------------------------------------------------

import {
  API_BASE,
  SessionError,
  clearSession,
  ensureSession,
  getSessionToken,
} from "@/lib/session";

/** Typed error so callers can distinguish API errors from programming errors. */
export class ApiError extends Error {
  constructor(
    message: string,
    /** HTTP status code, or 0 for network-level failures */
    public readonly status: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ── POST /chat ───────────────────────────────────────────────────────────────

/** A source document returned by POST /chat */
export interface ChatApiSource {
  filename: string;
  citation?: number;
  title?: string | null;
  url?: string | null;
  institution?: string | null;
  evidence_level?: string | null;
  chunk_id?: string | null;
  /** Absent in backend responses — see ChatSource in src/types/chat.ts. */
  match_score?: number;
}

/** One knowledge-graph relationship. */
export interface KnowledgeGraphEdge {
  subject: string;
  relation: string;
  object: string;
}

/** Shape returned by POST /chat */
export interface ChatApiResponse {
  user_message: string;
  reply: string;
  mode: string;
  sources?: ChatApiSource[];
  knowledge_graph?: KnowledgeGraphEdge[];
  /**
   * Present when the exchange was persisted for the current session. Clients
   * that ignore them keep working exactly as before.
   */
  conversation_id?: string;
  user_message_id?: string;
  assistant_message_id?: string;
}

// ── Conversation history ─────────────────────────────────────────────────────

/** The image stored with a user turn. */
export interface ConversationImage {
  id: string;
  /** Backend path serving the bytes, e.g. `/images/<id>`. */
  url: string;
  mime_type: string;
  byte_size: number;
}

/** A citation stored with an assistant turn. */
export interface MessageSource {
  /** Citation number used in the reply, e.g. `[1]` for `citation: 1`. */
  citation: number;
  filename: string | null;
  title: string | null;
  url: string | null;
  institution: string | null;
  evidence_level: string | null;
  chunk_id: string;
}

/** One stored message of a conversation. */
export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  /** Position within the conversation, starting at 1. */
  sequence: number;
  content: string;
  created_at: string;
  /** User turns: the attached image, when one was stored. */
  image?: ConversationImage;
  /** Assistant turns: pipeline outcome, e.g. "grounded_rag". */
  mode?: string | null;
  vision_assessment?: string | null;
  grounded_answer?: string | null;
  sources?: MessageSource[];
  knowledge_graph?: KnowledgeGraphEdge[];
}

/** Fields shared by every conversation response. */
export interface ConversationHeader {
  id: string;
  title: string;
  crop: string | null;
  created_at: string;
  updated_at: string;
}

/** One row of GET /conversations. */
export interface ConversationSummary extends ConversationHeader {
  message_count: number;
  /** True when a stored message carries a vision assessment. */
  has_diagnosis: boolean;
}

/** Shape returned by GET /conversations/{id}. */
export interface ConversationDetail extends ConversationHeader {
  messages: ConversationMessage[];
}

// ── GET /health ──────────────────────────────────────────────────────────────

/** Shape returned by GET /health */
export interface HealthResponse {
  status: string;
  service: string;
}

// ── Shared request plumbing ──────────────────────────────────────────────────

interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  /** Request body. FormData is left untouched so the browser sets the boundary. */
  body?: BodyInit;
  /** Extra headers. Callers must not set Content-Type for FormData. */
  headers?: Record<string, string>;
}

/** Fetch with the session token attached. */
async function send(
  path: string,
  options: RequestOptions,
  token: string
): Promise<Response> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...options.headers,
    "X-Session-Token": token,
  };

  try {
    return await fetch(`${API_BASE}${path}`, {
      method: options.method ?? "GET",
      headers,
      body: options.body,
    });
  } catch {
    throw new ApiError(
      `Unable to reach the CropSense AI backend. ` +
        `Please make sure the server is running at ${API_BASE}.`,
      0
    );
  }
}

/** Obtain a token, reporting a bootstrap failure as an ApiError. */
async function requireSessionToken(): Promise<string> {
  try {
    return await ensureSession();
  } catch (error) {
    if (error instanceof SessionError) {
      throw new ApiError(error.message, error.status);
    }
    throw new ApiError(
      "Unable to start a session with the server. Please try again.",
      0
    );
  }
}

/**
 * Turn a failed response into an ApiError, preferring the backend's own
 * `detail` message and falling back to the status line.
 */
async function toApiError(response: Response): Promise<ApiError> {
  let detail = `Server responded with ${response.status} ${response.statusText}.`;

  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const value = (body as { detail?: unknown }).detail;
      if (typeof value === "string" && value) detail = value;
    }
  } catch {
    // Not JSON (or an empty body): keep the status-line message.
  }

  return new ApiError(detail, response.status);
}

/**
 * The session refresh that is currently in flight, if any.
 *
 * Several requests can hit a 401 with the same expired token at the same time.
 * Each of them clearing the session and creating a new one would leave the user
 * with one anonymous session per request, so the replacement is shared here:
 * the first caller starts it and everyone else awaits the same promise.
 */
let refreshPromise: Promise<string> | null = null;

/**
 * Replace a token the server rejected, returning a token to retry with.
 *
 * - If another request already replaced `staleToken`, its token is reused
 *   instead of refreshing again.
 * - Otherwise exactly one POST /sessions is made, however many requests arrive
 *   with the same stale token.
 *
 * The cached promise is dropped once it settles, so a later 401 (or a failed
 * refresh) can start a fresh attempt.
 */
function refreshSession(staleToken: string): Promise<string> {
  const current = getSessionToken();
  if (current && current !== staleToken) {
    // The token was already replaced while this request was in flight.
    return Promise.resolve(current);
  }

  if (!refreshPromise) {
    // Cleared once, by whichever request owns the refresh, so the concurrent
    // callers below fall through to the shared creation instead of starting
    // their own.
    clearSession();
    refreshPromise = requireSessionToken().finally(() => {
      refreshPromise = null;
    });
  }

  return refreshPromise;
}

/**
 * Authenticated request helper.
 *
 * A 401 means the stored token is unknown or expired: one refresh is shared by
 * every request that hit the same token, and this request is retried exactly
 * once with the resulting token. A second 401 is reported to the caller, so the
 * retry can never loop.
 *
 * The body is reused for the retry; FormData is not a stream, so sending it
 * twice is safe.
 */
async function request(
  path: string,
  options: RequestOptions = {}
): Promise<Response> {
  const token = await requireSessionToken();
  let response = await send(path, options, token);

  if (response.status === 401) {
    const refreshed = await refreshSession(token);
    response = await send(path, options, refreshed);
  }

  if (!response.ok) {
    throw await toApiError(response);
  }

  return response;
}

/** Read a JSON body, reporting an unreadable one as an ApiError. */
async function readJson<T>(response: Response): Promise<T> {
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(
      "The server returned a response that could not be read.",
      response.status
    );
  }
}

/** Path segment for an id supplied by the caller. */
function conversationPath(id: string): string {
  return `/conversations/${encodeURIComponent(id)}`;
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Sends a chat message to the backend's POST /chat endpoint.
 *
 * - Text-only: sends `message` as a multipart form field (no image field).
 * - With image: sends `message` + `image` as multipart/form-data so the
 *   backend can read the raw bytes and pass them to the Gemini vision model.
 * - Continuing a conversation: adds `conversation_id` to the form.
 *
 * Using multipart for both cases keeps the backend interface uniform.
 *
 * @throws {ApiError} on network failure or non-2xx HTTP status
 */
export async function sendChatMessage(
  message: string,
  imageFile?: File,
  conversationId?: string
): Promise<ChatApiResponse> {
  const formData = new FormData();
  formData.append("message", message);

  if (imageFile) {
    // Validate MIME type and size on the client side before sending
    if (!imageFile.type.startsWith("image/")) {
      throw new ApiError("Only image files are supported.", 0);
    }
    if (imageFile.size > 10 * 1024 * 1024) {
      throw new ApiError("Image must be smaller than 10 MB.", 0);
    }
    formData.append("image", imageFile, imageFile.name);
  }

  if (conversationId) {
    formData.append("conversation_id", conversationId);
  }

  // Content-Type is deliberately not set: the browser adds the multipart
  // boundary itself when the body is FormData.
  const response = await request("/chat", { method: "POST", body: formData });
  return readJson<ChatApiResponse>(response);
}

/**
 * The current session's conversations, most recently updated first.
 *
 * @throws {ApiError} on network failure or non-2xx HTTP status
 */
export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await request("/conversations");
  return readJson<ConversationSummary[]>(response);
}

/**
 * One conversation with its ordered messages, citations and graph edges.
 * A conversation owned by another session is reported as 404.
 *
 * @throws {ApiError} on network failure or non-2xx HTTP status
 */
export async function getConversation(id: string): Promise<ConversationDetail> {
  const response = await request(conversationPath(id));
  return readJson<ConversationDetail>(response);
}

/**
 * Rename a conversation. The backend answers with the conversation header, so
 * fetch the history separately if the messages are needed.
 *
 * @throws {ApiError} on network failure, an invalid title (400/422) or an
 *   unknown or foreign conversation (404)
 */
export async function renameConversation(
  id: string,
  title: string
): Promise<ConversationHeader> {
  const response = await request(conversationPath(id), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return readJson<ConversationHeader>(response);
}

/**
 * Permanently delete a conversation, its messages and any unreferenced images.
 *
 * @throws {ApiError} on network failure or an unknown or foreign conversation
 *   (404)
 */
export async function deleteConversation(id: string): Promise<void> {
  await request(conversationPath(id), { method: "DELETE" });
}

/**
 * Download one stored image's bytes.
 *
 * GET /images/{id} requires X-Session-Token, so an `<img src>` cannot be used:
 * the component fetches the blob through this helper and renders a temporary
 * object URL instead. The token stays in the request header — never in a URL,
 * an attribute or the DOM.
 *
 * @throws {ApiError} on network failure or a missing or foreign image (404)
 */
export async function fetchImage(imageId: string): Promise<Blob> {
  const response = await request(`/images/${encodeURIComponent(imageId)}`, {
    headers: { Accept: "image/*" },
  });
  return response.blob();
}

/**
 * Calls GET /health and returns true if the backend is reachable and healthy.
 * Never throws — returns false on any failure so callers can degrade gracefully.
 *
 * Health is public, so this call carries no session token and cannot create one.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      // Short timeout via AbortController so the UI doesn't hang on load
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return false;
    const body: HealthResponse = await res.json();
    return body.status === "healthy";
  } catch {
    return false;
  }
}
