import type { NextConfig } from "next";

/**
 * `/api/*` is proxied to Django.
 *
 * WHY THIS EXISTS. Next serves the site on :3000 and Django serves the API on
 * :8000. If the browser called :8000 directly those are two different
 * ORIGINS, and the browser would then demand CORS headers on every API
 * response and refuse to send session cookies without extra configuration on
 * both sides — real work, and work that has to be undone later.
 *
 * In production Caddy already solves this by serving both from one domain
 * (SYSTEM_DESIGN §2.2). This rewrite makes local development behave the SAME
 * WAY: the browser only ever talks to :3000, Next quietly forwards anything
 * under /api/ to Django, and as far as the browser is concerned there is one
 * origin. So there is no CORS locally, none in production, and no behavioural
 * difference between the two to debug later.
 *
 * Set DJANGO_ORIGIN if Django is not on :8000.
 */
const DJANGO_ORIGIN = process.env.DJANGO_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,

  // `localhost:3000` and `127.0.0.1:3000` are the SAME server but DIFFERENT
  // origins to a browser, and Next 16 blocks its dev-only endpoints
  // (`/_next/webpack-hmr`, HMR chunks) when the request's origin is not on
  // this list.
  //
  // The failure is much worse than "hot reload stops": the dev client cannot
  // start, so React never hydrates, so NO effect and NO event handler ever
  // runs. The page still renders — Next served the HTML fine — it is just
  // completely inert. That presents as `RequireAuth` sitting on "Checking
  // your session…" forever (its redirect is an effect) and as buttons that
  // do nothing when clicked, neither of which points at the dev server.
  //
  // The only visible clue is a warning in the terminal, not the browser.
  // DEV ONLY — Next ignores this in a production build.
  allowedDevOrigins: ["localhost", "127.0.0.1", "[::1]"],

  // REQUIRED for the rewrite below to work at all.
  //
  // Django URLs end in a slash (`/api/health/`) and Django redirects to add
  // one if missing. Next does the exact opposite by default: it 308-redirects
  // `/api/health/` -> `/api/health` BEFORE any rewrite runs. The two
  // conventions fight, and the visible symptom is that every API call returns
  // a 308 with the path as its body instead of reaching Django at all.
  //
  // This turns Next's half of that off and lets the path through untouched.
  skipTrailingSlashRedirect: true,

  async rewrites() {
    return [
      {
        // allauth headless FIRST, and WITHOUT a trailing slash.
        //
        // Its routes are `/api/auth/browser/v1/config` etc. with no trailing
        // slash, unlike every route we wrote ourselves. Falling through to
        // the rule below would append one and 404 the entire auth surface —
        // which looks like "allauth isn't mounted" rather than a routing
        // bug. More specific rule, so it must come first.
        source: "/api/auth/:path*",
        destination: `${DJANGO_ORIGIN}/api/auth/:path*`,
      },
      {
        // Our own REST endpoints plus the Django admin (mounted at
        // /api/admin/ so this one rule covers it).
        //
        // NOTE THE TRAILING SLASH ON THE DESTINATION. Next drops it when it
        // reconstructs the path from `:path*`, Django's APPEND_SLASH then
        // 301s to add it back, and the result is an infinite redirect
        // between the two frameworks' conventions — which presents as
        // "every API call returns 301" with a `location` identical to the
        // URL you asked for.
        source: "/api/:path*",
        destination: `${DJANGO_ORIGIN}/api/:path*/`,
      },
      {
        // Django's own static files, so the admin and DRF's browsable API
        // render with styling instead of as bare unstyled HTML.
        source: "/static/:path*",
        destination: `${DJANGO_ORIGIN}/static/:path*`,
      },
    ];
  },
};

export default nextConfig;
