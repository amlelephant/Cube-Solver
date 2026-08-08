"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { api, ApiError } from "@/lib/api";
import { clearMe } from "@/lib/account";

/**
 * Session state, backed by allauth headless (`/api/auth/browser/v1/*`).
 *
 * Two allauth behaviours are worth knowing before reading this, because both
 * look like bugs:
 *
 *  * `GET auth/session` answers **401 when nobody is signed in**. That is the
 *    normal, expected answer — not an error to surface.
 *  * `POST auth/signup` also answers **401**, with `meta.is_authenticated:
 *    false` and a pending `verify_email` flow, because email verification is
 *    mandatory server-side. Signing up deliberately does NOT sign you in.
 */

const BASE = "/api/auth/browser/v1";

export type User = { id?: number; email?: string; username?: string } | null;

type AuthValue = {
  user: User;
  /** null while the first session check is in flight — render nothing decisive until it resolves. */
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<{ verificationRequired: boolean }>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
  /** Confirm the key from a verification email. Resolves once signed in. */
  verifyEmail: (key: string) => Promise<void>;
  /** Re-send the verification mail for an address mid-signup. */
  resendVerification: (email: string) => Promise<void>;
  /** Start "forgot password". Deliberately cannot report whether the address exists. */
  requestPasswordReset: (email: string) => Promise<void>;
  /** Finish "forgot password" with the key from the mail. */
  resetPassword: (key: string, password: string) => Promise<{ signedIn: boolean }>;
};

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await api<any>(`${BASE}/auth/session`);
      setUser(res?.data?.user ?? null);
    } catch (e) {
      // 401 here means "signed out", which is a state and not a failure.
      if (!(e instanceof ApiError) || (e.status !== 401 && e.status !== 410)) {
        console.error("session check failed", e);
      }
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const login = useCallback(
    async (email: string, password: string) => {
      const res = await api<any>(`${BASE}/auth/login`, {
        method: "POST",
        body: { email, password },
      });
      setUser(res?.data?.user ?? null);
      await refresh();
    },
    [refresh],
  );

  const signup = useCallback(async (email: string, password: string) => {
    try {
      const res = await api<any>(`${BASE}/auth/signup`, {
        method: "POST",
        body: { email, password },
      });
      setUser(res?.data?.user ?? null);
      return { verificationRequired: false };
    } catch (e) {
      // The expected success path: account created, not signed in, waiting on
      // email verification. Treating this 401 as a failure would tell the user
      // their signup broke when it worked.
      if (e instanceof ApiError && e.status === 401) {
        const flows = e.data?.data?.flows ?? [];
        const pending = flows.some(
          (f: any) => f.id === "verify_email" && f.is_pending,
        );
        if (pending) return { verificationRequired: true };
      }
      throw e;
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await api(`${BASE}/auth/session`, { method: "DELETE" });
    } catch (e) {
      // allauth answers 401 to a logout that already succeeded. Either way
      // the session is gone, so this must not block clearing local state.
      if (!(e instanceof ApiError) || e.status !== 401) throw e;
    }
    setUser(null);
    // The account store is module-level and outlives this provider's state,
    // so without this the next sign-in flashes the previous user's name and
    // avatar in the NavBar before `/api/me/` answers.
    clearMe();
  }, []);

  /**
   * allauth answers a SUCCESSFUL verification with 200 and a session — the
   * account is now usable and the visitor is signed in. It answers a spent or
   * mistyped key with 400, and a key for an account that still needs more
   * steps with 401; only the first of those is an error worth showing as one,
   * so let the caller see the status via ApiError rather than flattening it.
   */
  const verifyEmail = useCallback(
    async (key: string) => {
      await api(`${BASE}/auth/email/verify`, { method: "POST", body: { key } });
      await refresh();
    },
    [refresh],
  );

  const resendVerification = useCallback(async (email: string) => {
    await api(`${BASE}/auth/email/verify/resend`, {
      method: "POST",
      body: { email },
    });
  }, []);

  /**
   * Answers the same way whether or not the address has an account — that is
   * allauth's behaviour and it is the correct one. Reporting "no such user"
   * here would turn the reset form into an account-enumeration oracle, which
   * on a leaderboard product hands an attacker the list of real competitors.
   * So the UI must not promise the mail was sent, only that it was requested.
   */
  const requestPasswordReset = useCallback(async (email: string) => {
    await api(`${BASE}/auth/password/request`, {
      method: "POST",
      body: { email },
    });
  }, []);

  /**
   * Returns whether the reset also opened a session.
   *
   * THE 401 HERE IS A SUCCESS, exactly like signup's. allauth's
   * ACCOUNT_LOGIN_ON_PASSWORD_RESET defaults to False — deliberately, since
   * anyone holding the emailed key would otherwise get a live session — so a
   * completed reset answers 401 with `meta.is_authenticated: false`. Treating
   * that as a failure tells someone their reset broke immediately after it
   * worked, and sends them back to request another key they do not need.
   *
   * A genuinely bad key answers 400, which is left to throw.
   */
  const resetPassword = useCallback(
    async (key: string, password: string): Promise<{ signedIn: boolean }> => {
      try {
        await api(`${BASE}/auth/password/reset`, {
          method: "POST",
          body: { key, password },
        });
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) {
          const authed = e.data?.meta?.is_authenticated;
          if (authed === false) return { signedIn: false };
        }
        throw e;
      }
      await refresh();
      return { signedIn: true };
    },
    [refresh],
  );

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        signup,
        logout,
        refresh,
        verifyEmail,
        resendVerification,
        requestPasswordReset,
        resetPassword,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
