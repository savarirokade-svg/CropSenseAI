// ---------------------------------------------------------------------------
// CropSense AI — anonymous session
//
// The backend has no accounts. An anonymous session is created with
// POST /sessions and identified afterwards by the opaque token sent in the
// X-Session-Token header. Only that token is kept in the browser — no personal
// data, passwords or profile information is stored.
//
// This module imports nothing from ./api: api.ts imports the session helpers,
// so the dependency has to point one way only. That is why the backend base URL
// lives here — both modules need it, and this is the side api.ts may import.
// ---------------------------------------------------------------------------

/** localStorage key holding the anonymous session token. */
export const SESSION_STORAGE_KEY = "cropsense.session";

/**
 * Base URL for the FastAPI backend.
 *
 * Reads NEXT_PUBLIC_API_URL at build time so the frontend can target any
 * environment without a code change. The default keeps local dev working out of
 * the box.
 */
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/**
 * Raised when a session cannot be created.
 *
 * It never reaches callers of ./api: the request helper converts it into an
 * `ApiError`, so the rest of the app only has one error type to handle.
 */
export class SessionError extends Error {
  constructor(
    message: string,
    /** HTTP status code, or 0 for network-level failures */
    public readonly status: number
  ) {
    super(message);
    this.name = "SessionError";
  }
}

/** Only `token` is used here; the rest of POST /sessions is informational. */
interface SessionResponse {
  token: string;
  session_id: string;
  created_at: string;
  expires_at: string;
}

/**
 * In-memory copy of the token.
 *
 * It is the fallback for browsers where localStorage is unavailable (private
 * mode, storage disabled) and it means a single page view keeps working even
 * when writing to storage fails. It is never shared between visitors because
 * sessions are only created in the browser — see ensureSession().
 */
let memoryToken: string | null = null;

/** The in-flight creation, so concurrent callers share one POST /sessions. */
let pendingSession: Promise<string> | null = null;

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

/** localStorage when it exists and is usable, otherwise null. */
function getStorage(): Storage | null {
  if (!isBrowser()) return null;
  try {
    return window.localStorage;
  } catch {
    // Merely touching localStorage throws when storage is blocked.
    return null;
  }
}

/**
 * The stored session token, without touching the network.
 *
 * Returns null on the server and when no session has been created yet, so
 * callers can decide whether to bootstrap one.
 */
export function getSessionToken(): string | null {
  if (memoryToken) return memoryToken;

  const storage = getStorage();
  if (!storage) return null;

  try {
    const token = storage.getItem(SESSION_STORAGE_KEY);
    return token && token.trim() ? token : null;
  } catch {
    return null;
  }
}

function storeSessionToken(token: string): void {
  memoryToken = token;

  const storage = getStorage();
  if (!storage) return;

  try {
    storage.setItem(SESSION_STORAGE_KEY, token);
  } catch {
    // Quota exceeded or storage disabled: the in-memory copy still works.
  }
}

/**
 * Forget the stored token.
 *
 * The next authenticated call boots a new session. Also drops the cached
 * creation promise so a retry after an unexpected 401 cannot reuse it.
 */
export function clearSession(): void {
  memoryToken = null;
  pendingSession = null;

  const storage = getStorage();
  if (!storage) return;

  try {
    storage.removeItem(SESSION_STORAGE_KEY);
  } catch {
    // Nothing else to clean up.
  }
}

/** POST /sessions and return the new raw token. */
async function createSession(): Promise<string> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/sessions`, {
      method: "POST",
      headers: { Accept: "application/json" },
    });
  } catch {
    throw new SessionError(
      `Unable to reach the CropSense AI backend. ` +
        `Please make sure the server is running at ${API_BASE}.`,
      0
    );
  }

  if (!response.ok) {
    throw new SessionError(
      `The server could not start a session (${response.status} ${response.statusText}).`,
      response.status
    );
  }

  let body: SessionResponse;
  try {
    body = (await response.json()) as SessionResponse;
  } catch {
    throw new SessionError(
      "The server returned an unreadable session response.",
      response.status
    );
  }

  if (!body || typeof body.token !== "string" || !body.token) {
    throw new SessionError(
      "The server returned an invalid session response.",
      response.status
    );
  }

  return body.token;
}

/**
 * Return the session token, creating a session on first use.
 *
 * A stored token is reused as-is. Otherwise one POST /sessions is made and
 * shared by every caller that arrives while it is in flight, so a burst of
 * requests can never create several sessions. The promise is dropped once it
 * settles: a later call reads the stored token instead.
 *
 * @throws {SessionError} when no session can be created (mapped to ApiError by
 *   ./api, so UI code keeps handling a single error type)
 */
export async function ensureSession(): Promise<string> {
  const existing = getSessionToken();
  if (existing) return existing;

  if (!isBrowser()) {
    // Sessions are browser-scoped. Creating one during server rendering would
    // hand the same token to every visitor, so it is refused instead.
    throw new SessionError("Sessions are only available in the browser.", 0);
  }

  if (!pendingSession) {
    pendingSession = createSession()
      .then((token) => {
        storeSessionToken(token);
        return token;
      })
      .finally(() => {
        pendingSession = null;
      });
  }

  return pendingSession;
}
