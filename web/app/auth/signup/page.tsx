import { redirect } from "next/navigation";

/**
 * `/auth/signup` exists because settings.py's HEADLESS_FRONTEND_URLS points
 * allauth's own mail at it — a link in a real email that 404s is worse than
 * one that goes somewhere plain.
 *
 * Sign-up is the login screen in its other mode rather than a second copy of
 * the same two fields, so this route just carries you there. It forwards
 * `?next=` so a deep link still survives.
 */
export default async function SignupPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const raw = params.next;
  const next = Array.isArray(raw) ? raw[0] : raw;

  // Only ever forward a same-site path. An attacker-supplied `?next=` is an
  // open-redirect otherwise, and the login page will happily route to it
  // after a successful sign-in.
  const safe = next && next.startsWith("/") && !next.startsWith("//") ? next : null;

  redirect(
    `/auth/login?mode=signup${safe ? `&next=${encodeURIComponent(safe)}` : ""}`,
  );
}
