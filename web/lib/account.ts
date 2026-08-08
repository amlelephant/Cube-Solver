"use client";

import { useEffect, useSyncExternalStore } from "react";
import { api, ApiError } from "@/lib/api";

/**
 * The account/profile API.
 *
 * `Me` and `PublicProfile` are deliberately different types even though the
 * server builds both from one function: `Me` carries the email and the
 * rate-limit clocks, and `PublicProfile` must never grow them. Keeping them
 * apart in the type system means an accidental `profile.email` on a page
 * showing someone ELSE's profile is a compile error rather than a leak.
 */

export type SolveRow = {
  id: number;
  seconds: number;
  result: "win" | "loss" | "solo";
  opponent: string | null;
  verdict: "verified" | "rejected" | "review";
  rating_delta: number | null;
  received_at: string;
};

export type PublicProfile = {
  id: number;
  username: string;
  country: string | null;
  rating: number;
  rank?: number;
  best_rank: number | null;
  is_founder: boolean;
  /** Preset avatar key, or null. Validated server-side; see lib/avatars.tsx. */
  avatar: string | null;
  best_seconds: number | null;
  total_solves: number;
  /**
   * The fastest verified solve, so "Current Best" can link to it. Absent on
   * leaderboard rows, which build their stats without it — hence optional
   * rather than nullable.
   */
  best_solve_id?: number | null;
  joined: string;
  solves?: SolveRow[];
};

export type Me = PublicProfile & {
  email: string;
  /** The paid Coach tier. Private — never on PublicProfile. */
  is_premium: boolean;
  notify_invites: boolean;
  notify_recap: boolean;
  notify_pb: boolean;
  /** Seconds until each change is allowed again. 0 means now. */
  limits: { username: number; email: number; password: number };
};

export const getMe = () => api<Me>("/api/me/");

export const patchMe = (body: Partial<
  Pick<Me, "country" | "avatar" | "notify_invites" | "notify_recap" | "notify_pb">
>) => api<Me>("/api/me/", { method: "PATCH", body });

export const changeUsername = (username: string) =>
  api<{ username: string; retry_after: number }>("/api/me/username/", {
    method: "POST",
    body: { username },
  });

export const changeEmail = (email: string) =>
  api<{ pending_email: string; message: string; retry_after: number }>(
    "/api/me/email/",
    { method: "POST", body: { email } },
  );

/**
 * Password change is allauth's endpoint, not ours — so this is the one call
 * here that does not go through /api/me/. The daily limit on it is enforced
 * by middleware server-side (see server/core/middleware.py), which is why a
 * 429 can come back from an endpoint we did not write.
 */
export const changePassword = (current: string, next: string) =>
  api("/api/auth/browser/v1/account/password/change", {
    method: "POST",
    body: { current_password: current, new_password: next },
  });

export const getProfile = (username: string) =>
  api<PublicProfile>(`/api/users/${encodeURIComponent(username)}/`);

export const getLeaderboard = (limit = 25) =>
  api<{ results: PublicProfile[] }>(`/api/leaderboard/?limit=${limit}`);

/** Turn a `retry_after` in seconds into something a person can read. */
export function humanDelay(seconds: number): string {
  if (seconds <= 0) return "now";
  const d = Math.floor(seconds / 86400);
  if (d >= 1) return d === 1 ? "in 1 day" : `in ${d} days`;
  const h = Math.floor(seconds / 3600);
  if (h >= 1) return h === 1 ? "in 1 hour" : `in ${h} hours`;
  const m = Math.max(1, Math.floor(seconds / 60));
  return m === 1 ? "in 1 minute" : `in ${m} minutes`;
}

/**
 * The signed-in account, shared by every component that asks for it.
 *
 * ONE STORE, NOT ONE PER CALLER. This was `useState` inside the hook, which
 * gave each of the six call sites its own copy and its own `/api/me/`
 * request. Two consequences, both real: the NavBar and the profile page
 * disagreed about your avatar the moment you changed it — the picker's
 * `setMe` updated the picker's copy and nothing else — and every page made
 * three identical requests for the same object.
 *
 * `useSyncExternalStore` rather than a context provider because there is
 * exactly one account per session: a provider would need to wrap the tree
 * and would still be a singleton in practice.
 */

type MeState = { me: Me | null; loading: boolean; error: string | null };

let state: MeState = { me: null, loading: true, error: null };
const listeners = new Set<() => void>();
/** In-flight request, so N mounting components produce ONE fetch. */
let inFlight: Promise<void> | null = null;

function emit(next: Partial<MeState>) {
  // A fresh object each time: `useSyncExternalStore` compares snapshots by
  // identity, so mutating in place would render nothing.
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Fetch the account. Concurrent callers share the one request. */
export function reloadMe(): Promise<void> {
  if (inFlight) return inFlight;
  inFlight = getMe()
    .then((me) => emit({ me, error: null, loading: false }))
    .catch((e) => {
      // 401/403 is "signed out", which RequireAuth is already handling — do
      // not also paint an error over a page that is about to be replaced.
      if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
        emit({ me: null, error: null, loading: false });
      } else {
        emit({
          error: (e as Error)?.message ?? "Could not load your account.",
          loading: false,
        });
      }
    })
    .finally(() => {
      inFlight = null;
    });
  return inFlight;
}

/** Replace the cached account — every consumer re-renders. */
export function setMeShared(me: Me | null) {
  emit({ me });
}

/** Drop the cached account. Call on sign-out so the next session does not
 *  briefly render the previous user's name. */
export function clearMe() {
  state = { me: null, loading: true, error: null };
  listeners.forEach((l) => l());
}

const getSnapshot = () => state;

export function useMe() {
  const snapshot = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  useEffect(() => {
    // Only the first mount fetches; later mounts read what is already here.
    if (state.me === null && state.error === null) void reloadMe();
  }, []);

  return {
    me: snapshot.me,
    setMe: setMeShared,
    loading: snapshot.loading,
    error: snapshot.error,
    reload: reloadMe,
  };
}
