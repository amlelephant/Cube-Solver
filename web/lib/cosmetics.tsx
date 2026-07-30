"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { WreathTier } from "@/components/LaurelWreath";
import { you } from "@/lib/mockData";

const STORAGE_KEY = "cubearena-wreath";

/** The rank-earned tiers, best first — the only ones the picker ever lists. */
export const RANKED_TIERS = ["gold", "silver", "bronze"] as const satisfies WreathTier[];

const ALL_TIERS = [...RANKED_TIERS, "founder"] as const satisfies WreathTier[];

/**
 * The stored tier is checked against what the account has actually earned, not
 * just against the set of valid tier names. Without this, editing localStorage
 * equips anything — which for the ranked tiers is only vanity, but would hand
 * anyone the secret founder's olive and then name it back to them in the
 * "Remove ..." control. Client-side this is still only a speed bump; the real
 * guarantee has to come from the server that owns `isFounder`.
 */
function isEarned(tier: WreathTier) {
  return earnedWreaths(you.bestRank, you.isFounder).includes(tier);
}

type CosmeticsContextValue = {
  equippedWreath: WreathTier | null;
  setEquippedWreath: (tier: WreathTier | null) => void;
};

const CosmeticsContext = createContext<CosmeticsContextValue | null>(null);

export function CosmeticsProvider({ children }: { children: ReactNode }) {
  const [equippedWreath, setEquippedState] = useState<WreathTier | null>(null);

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored && (ALL_TIERS as string[]).includes(stored) && isEarned(stored as WreathTier)) {
      setEquippedState(stored as WreathTier);
    } else if (stored) {
      // A tier this account has no claim to — drop it rather than wear it.
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const setEquippedWreath = (tier: WreathTier | null) => {
    if (tier && !isEarned(tier)) return;
    setEquippedState(tier);
    if (tier) {
      window.localStorage.setItem(STORAGE_KEY, tier);
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  };

  return (
    <CosmeticsContext.Provider value={{ equippedWreath, setEquippedWreath }}>
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
