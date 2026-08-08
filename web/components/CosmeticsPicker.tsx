"use client";

import { useState } from "react";
import { Check, Lock } from "lucide-react";
import { Avatar } from "@/components/Avatar";
import { TIER_LABEL, type WreathTier } from "@/components/LaurelWreath";
import { cn } from "@/lib/cn";
import { useCosmetics, RANKED_TIERS } from "@/lib/cosmetics";
import { AVATAR_PRESETS } from "@/lib/avatars";
import { patchMe, useMe } from "@/lib/account";

/**
 * Your avatar and your wreath, in one sheet.
 *
 * The two are stored very differently and the split is deliberate. The
 * WREATH is a local preference over a server-granted entitlement — what you
 * may wear comes from `/api/me/`, what you have equipped is localStorage.
 * The AVATAR is server state: it appears on your public profile, so other
 * people's browsers have to be able to read it, which localStorage cannot do.
 *
 * There are no uploads. Image moderation — review, reporting, takedown — is
 * not built, and a public profile is exactly where unmoderated user imagery
 * does damage. Presets need none of it: the server stores a key from a fixed
 * allowlist and the artwork is drawn client-side.
 */

const WREATH_REQUIREMENT: Record<WreathTier, string> = {
  gold: "Reach #1 on the global leaderboard",
  silver: "Reach the top 2",
  bronze: "Reach the top 3",
  founder: "Granted, not earned",
};

export function CosmeticsPicker() {
  // `earned` is the SERVER's answer, carried by the provider — never a
  // client-side literal, which is what made the founder tier mintable before.
  const { equippedWreath, setEquippedWreath, earned } = useCosmetics();
  const { me, setMe } = useMe();
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The ranked three always show, locked or not — they tell you what there is
  // to chase. Anything else only appears once it is actually yours, which is
  // what makes the founder's olive a secret rather than a taunt.
  const tiers: WreathTier[] = [
    ...RANKED_TIERS,
    ...earned.filter((t) => !(RANKED_TIERS as readonly WreathTier[]).includes(t)),
  ];

  async function chooseAvatar(key: string) {
    if (!me || saving) return;
    const next = me.avatar === key ? "" : key;
    const previous = me.avatar;
    setMe({ ...me, avatar: next || null }); // optimistic — this is a tap target
    setSaving(key);
    setError(null);
    try {
      const updated = await patchMe({ avatar: next });
      setMe(updated);
    } catch (e) {
      setMe({ ...me, avatar: previous });
      setError((e as Error)?.message ?? "Could not save your avatar.");
    } finally {
      setSaving(null);
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-col gap-4">
        <div>
          <p className="text-sm font-medium">Avatar</p>
          <p className="mt-1 text-sm text-ink-faint">
            Pick one of these. Uploading your own is not available — we would
            need to moderate every image, and we would rather not ship that
            half-done.
          </p>
        </div>

        <div className="grid grid-cols-4 gap-3 sm:grid-cols-6">
          {AVATAR_PRESETS.map((preset) => {
            const chosen = me?.avatar === preset.key;
            return (
              <button
                key={preset.key}
                type="button"
                aria-pressed={chosen}
                aria-label={preset.label}
                title={preset.label}
                disabled={!me || saving !== null}
                onClick={() => chooseAvatar(preset.key)}
                className={cn(
                  "relative flex items-center justify-center rounded-xl border p-2 transition-colors",
                  // A border alone was almost invisible in dark mode, where
                  // `border-ink` sits a shade off the card it is drawn on.
                  // The ring plus the check badge read at a glance in both
                  // themes, which is what "which one am I wearing" needs.
                  chosen
                    ? "border-ink bg-cloud ring-2 ring-ink"
                    : "border-mist hover:bg-cloud/60",
                  saving === preset.key && "animate-pulse",
                  (!me || saving !== null) && !chosen && "opacity-60",
                )}
              >
                <Avatar size={44} preset={preset.key} />
                {chosen && (
                  <span
                    aria-hidden
                    className="absolute -right-1.5 -top-1.5 flex size-5 items-center justify-center rounded-full bg-ink text-paper"
                  >
                    <Check size={12} strokeWidth={3} />
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {error && <p className="text-xs text-cube-red">{error}</p>}

        {me?.avatar && (
          <button
            type="button"
            onClick={() => chooseAvatar(me.avatar!)}
            className="self-start text-xs font-medium text-ink-faint underline-offset-2 hover:text-ink hover:underline"
          >
            Remove avatar
          </button>
        )}
      </section>

      <section className="flex flex-col gap-4 border-t border-mist pt-6">
        <div>
          <p className="text-sm font-medium">Wreath</p>
          <p className="mt-1 text-sm text-ink-faint">
            Earned by your best-ever leaderboard rank, not bought.
            {me?.best_rank ? ` Your best-ever rank: #${me.best_rank}.` : ""}
          </p>
        </div>

        <div className="grid grid-cols-3 gap-3">
          {tiers.map((tier) => {
            const isEarned = earned.includes(tier);
            const isEquipped = equippedWreath === tier;
            return (
              <button
                key={tier}
                type="button"
                disabled={!isEarned}
                onClick={() => setEquippedWreath(isEquipped ? null : tier)}
                className={cn(
                  "flex flex-col items-center gap-3 rounded-xl border p-4 text-center transition-colors",
                  isEquipped ? "border-ink bg-cloud" : "border-mist hover:bg-cloud/60",
                  !isEarned && "cursor-not-allowed opacity-50 hover:bg-transparent",
                )}
              >
                <Avatar size={56} wreath={tier} preset={me?.avatar} />
                <div>
                  <p className="text-sm font-medium capitalize">{tier}</p>
                  <p className="mt-0.5 flex items-center justify-center gap-1 text-xs text-ink-faint">
                    {!isEarned && <Lock size={11} />}
                    {isEarned
                      ? isEquipped
                        ? "Equipped"
                        : "Tap to equip"
                      : WREATH_REQUIREMENT[tier]}
                  </p>
                </div>
              </button>
            );
          })}
        </div>

        {equippedWreath && (
          <button
            type="button"
            onClick={() => setEquippedWreath(null)}
            className="self-start text-xs font-medium text-ink-faint underline-offset-2 hover:text-ink hover:underline"
          >
            Remove {TIER_LABEL[equippedWreath].toLowerCase()}
          </button>
        )}
      </section>
    </div>
  );
}
