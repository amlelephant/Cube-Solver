/**
 * Thin fetch wrapper for the Django API.
 *
 * Everything goes to a RELATIVE path. Next rewrites `/api/*` to Django
 * (next.config.ts), so the browser only ever sees one origin — which is what
 * makes session cookies work without CORS. Never put an absolute
 * `http://localhost:8000` URL in here; it would work in dev and break the
 * moment cookies matter.
 */

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly data: any = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function getCookie(name: string): string | null {
  const match = document.cookie.match(
    new RegExp(`(?:^|; )${name.replace(/([.$?*|{}()[\]\\/+^])/g, "\\$1")}=([^;]*)`),
  );
  return match ? decodeURIComponent(match[1]) : null;
}

/**
 * Django requires a CSRF token on every unsafe request, and only sets the
 * cookie once something has asked for it. allauth's `config` endpoint does,
 * so this primes it.
 *
 * Skipping this is the single most common way to break an allauth headless
 * frontend: the request comes back as a bare `403` with an HTML error body,
 * which looks like an auth failure rather than a missing header.
 */
let csrfPrimed: Promise<void> | null = null;

export function primeCsrf(): Promise<void> {
  if (!csrfPrimed) {
    csrfPrimed = fetch("/api/auth/browser/v1/config", {
      credentials: "include",
    })
      .then(() => undefined)
      .catch(() => {
        // Let the caller's own request produce the real error rather than
        // failing here — a priming failure is almost always the API being
        // down, which the actual call will report far more usefully.
        csrfPrimed = null;
      });
  }
  return csrfPrimed;
}

type Options = Omit<RequestInit, "body"> & { body?: unknown };

export async function api<T = any>(path: string, opts: Options = {}): Promise<T> {
  const method = (opts.method ?? "GET").toUpperCase();
  const unsafe = !["GET", "HEAD", "OPTIONS", "TRACE"].includes(method);

  if (unsafe && !getCookie("csrftoken")) await primeCsrf();

  const headers = new Headers(opts.headers);
  if (opts.body !== undefined) headers.set("content-type", "application/json");
  if (unsafe) {
    const token = getCookie("csrftoken");
    if (token) headers.set("X-CSRFToken", token);
  }

  const res = await fetch(path, {
    ...opts,
    method,
    headers,
    credentials: "include", // send + accept the session cookie
    body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  });

  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    // Django renders HTML for 403/500. Surfacing the raw page would be
    // useless in a toast, so keep the status and a readable message.
    if (!res.ok) {
      throw new ApiError(
        res.status === 403
          ? "Request rejected (CSRF or permissions)."
          : `Server error (${res.status}).`,
        res.status,
      );
    }
  }

  if (!res.ok) {
    throw new ApiError(errorMessage(data, res.status), res.status, data);
  }
  return data as T;
}

/** Pull something human-readable out of an allauth or DRF error body. */
function errorMessage(data: any, status: number): string {
  if (data?.error) return String(data.error);
  // allauth: { errors: [{ message, param }] }
  if (Array.isArray(data?.errors) && data.errors.length) {
    return data.errors.map((e: any) => e.message).filter(Boolean).join(" ");
  }
  if (status === 401) return "Not signed in.";
  return `Something went wrong (${status}).`;
}
