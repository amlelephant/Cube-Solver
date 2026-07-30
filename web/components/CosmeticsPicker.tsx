"use client";

import { Lock } from "lucide-react";
import { Avatar } from "@/components/Avatar";
import { TIER_LABEL, type WreathTier } from "@/components/LaurelWreath";
import { cn } from "@/lib/cn";
import { useCosmetics, earnedWreaths, RANKED_TIERS } from "@/lib/cosmetics";
import { you } from "@/lib/mockData";

const WREATH_REQUIREMENT: Record<WreathTier, string> = {
  gold: "Reach #1 on the global leaderboard",
  silver: "Reach the top 2",
  bronze: "Reach the top 3",
  founder: "Granted, not earned",
};

export function CosmeticsPicker() {
  const { equippedWreath, setEquippedWreath } = useCosmetics();
  const earned = earnedWreaths(you.bestRank, you.isFounder);

  // The ranked three always show, locked or not — they tell you what there is
  // to chase. Anything else only appears once it is actually yours, which is
  // what makes the founder's olive a secret rather than a taunt.
  const tiers: WreathTier[] = [
    ...RANKED_TIERS,
    ...earned.filter((t) => !(RANKED_TIERS as readonly WreathTier[]).includes(t)),
  ];

  return (
    <div className="flex flex-col gap-5">
      <p className="text-sm text-ink-faint">
        Laurel wreaths are earned by your best-ever leaderboard rank, not bought. Your best-ever
        rank: #{you.bestRank}.
      </p>

      <div className="grid grid-cols-3 gap-3">
        {tiers.map((tier) => {
          const isEarned = earned.includes(tier);
          const isEquipped = equippedWreath === tier;
          return (
            <button
              key={tier}
              disabled={!isEarned}
              onClick={() => setEquippedWreath(isEquipped ? null : tier)}
              className={cn(
                "flex flex-col items-center gap-3 rounded-xl border p-4 text-center transition-colors",
                isEquipped ? "border-ink bg-cloud" : "border-mist hover:bg-cloud/60",
                !isEarned && "cursor-not-allowed opacity-50 hover:bg-transparent",
              )}
            >
              <Avatar size={56} wreath={tier} />
              <div>
                <p className="text-sm font-medium capitalize">{tier}</p>
                <p className="mt-0.5 flex items-center justify-center gap-1 text-xs text-ink-faint">
                  {!isEarned && <Lock size={11} />}
                  {isEarned ? (isEquipped ? "Equipped" : "Tap to equip") : WREATH_REQUIREMENT[tier]}
                </p>
              </div>
            </button>
          );
        })}
      </div>

      {equippedWreath && (
        <button
          onClick={() => setEquippedWreath(null)}
          className="self-start text-xs font-medium text-ink-faint underline-offset-2 hover:text-ink hover:underline"
        >
          Remove {TIER_LABEL[equippedWreath].toLowerCase()}
        </button>
      )}
    </div>
  );
}
