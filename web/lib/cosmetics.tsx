"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { WreathTier } from "@/components/LaurelWreath";
import { getMe } from "@/lib/account";

const STORAGE_KEY = "cubearena-wreath";

/** The rank-earned tiers, best first — the only ones the picker ever lists. */
export const RANKED_TIERS = ["gold", "silver", "bronze"] as const satisfies WreathTier[];

const ALL_TIERS = [...RANKED_TIERS, "founder"] as const satisfies WreathTier[];

type CosmeticsContextValue = {
  equippedWreath: WreathTier | null;
  setEquippedWreath: (tier: WreathTier | null) => void;
  /** What this account may wear, per the SERVER. Empty until /api/me/ answers. */
  earned: WreathTier[];
};

const CosmeticsContext = createContext<CosmeticsContextValue | null>(null);

/**
 * What is equipped is a local preference; what is EARNED comes from the
 * server.
 *
 * That split is the whole design. `best_rank` and `is_founder` used to be
 * literals in mock data, which meant the secret founder's olive was mintable
 * by anyone who edited a JS bundle — the previous version of this file said
 * so in a comment and could not do anything about it. Now the entitlement
 * arrives from `/api/me/`, and localStorage only ever chooses among the tiers
 * the server already granted.
 *
 * This is still a client-side check and still only a speed bump. It stops
 * being one when a wreath is rendered on someone ELSE'S screen — that comes
 * from their profile's `best_rank`/`is_founder`, which the client never
 * writes.
 */
export function CosmeticsProvider({ children }: { children: ReactNode }) {
  const [equippedWreath, setEquippedState] = useState<WreathTier | null>(null);
  const [earned, setEarned] = useState<WreathTier[]>([]);

  useEffect(() => {
    let cancelled = false;
    getMe()
      .then((me) => {
        if (cancelled) return;
        const allowed = earnedWreaths(me.best_rank ?? Infinity, me.is_founder);
        setEarned(allowed);

        const stored = window.localStorage.getItem(STORAGE_KEY);
        if (stored && allowed.includes(stored as WreathTier)) {
          setEquippedState(stored as WreathTier);
        } else if (stored) {
          // A tier this account has no claim to — drop it rather than wear it.
          window.localStorage.removeItem(STORAGE_KEY);
          setEquippedState(null);
        }
      })
      .catch(() => {
        // Signed out, or the API is down. Wear nothing rather than trusting
        // whatever localStorage happens to say.
        if (!cancelled) setEarned([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const setEquippedWreath = (tier: WreathTier | null) => {
    if (tier && !earned.includes(tier)) return;
    setEquippedState(tier);
    if (tier) {
      window.localStorage.setItem(STORAGE_KEY, tier);
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  };

  return (
    <CosmeticsContext.Provider value={{ equippedWreath, setEquippedWreath, earned }}>
      {children}
    </CosmeticsContext.Provider>
  );
}

export function useCosmetics() {
  const ctx = useContext(CosmeticsContext);
  if (!ctx) throw new Error("useCosmetics must be used within a CosmeticsProvider");
  return ctx;
}

/**
 * Ranks 1-3 earn every tier at or below their best-ever placement.
 *
 * `founder` is not rank-earned and is not merely *locked* for everyone else —
 * it is secret, so it is absent from this list rather than present-and-greyed.
 * The picker renders exactly what this returns plus the ranked tiers, which is
 * what keeps an unearned founder olive from being advertised to anyone who
 * cannot have one.
 */
export function earnedWreaths(bestRank: number, isFounder = false): WreathTier[] {
  const tiers: WreathTier[] = [];
  if (bestRank <= 3) tiers.push("bronze");
  if (bestRank <= 2) tiers.push("silver");
  if (bestRank <= 1) tiers.push("gold");
  if (isFounder) tiers.push("founder");
  return tiers;
}
