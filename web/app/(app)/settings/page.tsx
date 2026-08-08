"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Sun, Moon, Check } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { PasswordInput } from "@/components/ui/PasswordInput";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { Switch } from "@/components/ui/Switch";
import { CountryPicker } from "@/components/CountryPicker";
import { useTheme } from "@/lib/theme";
import { ApiError } from "@/lib/api";
import {
  changeEmail,
  changePassword,
  changeUsername,
  humanDelay,
  patchMe,
  useMe,
  type Me,
} from "@/lib/account";
import { cn } from "@/lib/cn";

function ComingSoonTag() {
  return (
    <span className="rounded-full bg-cloud px-2.5 py-1 text-xs font-medium text-ink-faint">
      Coming soon
    </span>
  );
}

function SettingsSection({
  title,
  description,
  tag,
  children,
}: {
  title: string;
  description: string;
  tag?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card className="p-8">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xl font-medium">{title}</p>
          <p className="mt-1 max-w-md text-sm text-ink-faint">{description}</p>
        </div>
        {tag}
      </div>
      <div className="mt-6 flex flex-col gap-5">{children}</div>
    </Card>
  );
}

function SettingsRow({
  label,
  description,
  control,
  children,
}: {
  label: string;
  description?: string;
  control?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="border-t border-mist pt-5 first:border-t-0 first:pt-0">
      <div className="flex items-center justify-between gap-6">
        <div>
          <p className="text-sm font-medium">{label}</p>
          {description && (
            <p className="mt-0.5 text-xs text-ink-faint">{description}</p>
          )}
        </div>
        {control}
      </div>
      {children}
    </div>
  );
}

const inputClass = cn(
  "w-full rounded-lg border border-mist bg-cloud px-3 py-2 text-sm text-ink",
  "placeholder:text-ink-faint outline-none transition-colors",
  "focus:border-ink/30 focus:bg-paper",
);

/** Inline result line for one of the account forms. */
function Feedback({ ok, error }: { ok?: string | null; error?: string | null }) {
  if (!ok && !error) return null;
  return (
    <p
      role="status"
      className={cn(
        "mt-2 flex items-center gap-1.5 text-xs",
        error ? "text-cube-red" : "text-cube-green",
      )}
    >
      {!error && <Check size={13} />}
      {error || ok}
    </p>
  );
}

/**
 * An account field that is rate limited. Renders its own form, its own
 * error/success line, and — when the limit is live — says WHEN it will be
 * available again rather than just refusing. A bare disabled control with no
 * explanation reads as a bug.
 */
function LimitedField({
  label,
  description,
  retryAfter,
  children,
}: {
  label: string;
  description: string;
  retryAfter: number;
  children: (locked: boolean) => ReactNode;
}) {
  const locked = retryAfter > 0;
  return (
    <SettingsRow
      label={label}
      description={
        locked ? `${description} Changeable again ${humanDelay(retryAfter)}.` : description
      }
    >
      <div className="mt-3">{children(locked)}</div>
    </SettingsRow>
  );
}

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const { me, setMe, loading, error: loadError } = useMe();

  if (loading) {
    return (
      <div className="mx-auto max-w-3xl px-6 pb-24 pt-16 md:px-10">
        <h1 className="text-5xl font-bold tracking-tight md:text-6xl">Settings</h1>
        <div className="mt-10 h-40 animate-pulse rounded-2xl bg-cloud" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl px-6 pb-24 pt-16 md:px-10">
      <h1 className="text-5xl font-bold tracking-tight md:text-6xl">Settings</h1>
      <p className="mt-3 text-lg text-ink-faint">
        Looking for avatar cosmetics? Click your avatar on your profile.
      </p>

      {loadError && (
        <p role="alert" className="mt-6 rounded-lg bg-cube-red/10 px-4 py-3 text-sm text-cube-red">
          {loadError}
        </p>
      )}

      <div className="mt-10 flex flex-col gap-6">
        <SettingsSection
          title="Appearance"
          description="Choose how CubeArena looks on this device."
        >
          <SettingsRow
            label="Theme"
            description="Switch between a light and dark interface."
            control={
              <SegmentedControl
                options={[
                  { value: "light" as const, label: "Light" },
                  { value: "dark" as const, label: "Dark" },
                ]}
                value={theme}
                onChange={setTheme}
              />
            }
          />
          <SettingsRow
            label="Preview"
            control={
              <div className="flex items-center gap-2 text-ink-faint">
                {theme === "light" ? <Sun size={18} /> : <Moon size={18} />}
                <span className="text-sm capitalize">{theme} mode</span>
              </div>
            }
          />
        </SettingsSection>

        {me && <AccountSection me={me} setMe={setMe} />}
        {me && <ProfileSection me={me} setMe={setMe} />}
        {me && <NotificationsSection me={me} setMe={setMe} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function AccountSection({ me, setMe }: { me: Me; setMe: (m: Me) => void }) {
  const [limits, setLimits] = useState(me.limits);
  useEffect(() => setLimits(me.limits), [me.limits]);

  return (
    <SettingsSection title="Account" description="Your sign-in details.">
      <SettingsRow
        label="User ID"
        description="Your account number — CubeArena members are numbered in the order they joined."
        control={
          <span className="rounded-lg bg-cloud px-4 py-2 font-mono text-sm font-semibold">
            #{me.id}
          </span>
        }
      />

      <UsernameField
        me={me}
        setMe={setMe}
        retryAfter={limits.username}
        onChanged={(retry) => setLimits({ ...limits, username: retry })}
      />
      <EmailField
        me={me}
        retryAfter={limits.email}
        onChanged={(retry) => setLimits({ ...limits, email: retry })}
      />
      <PasswordField
        retryAfter={limits.password}
        onChanged={() => setLimits({ ...limits, password: 24 * 3600 })}
      />
    </SettingsSection>
  );
}

function UsernameField({
  me,
  setMe,
  retryAfter,
  onChanged,
}: {
  me: Me;
  setMe: (m: Me) => void;
  retryAfter: number;
  onChanged: (retry: number) => void;
}) {
  const [value, setValue] = useState(me.username);
  const [busy, setBusy] = useState(false);
  const [ok, setOk] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setOk(null);
    setError(null);
    setBusy(true);
    try {
      const res = await changeUsername(value.trim());
      setMe({ ...me, username: res.username });
      onChanged(res.retry_after);
      setOk("Username updated.");
    } catch (e) {
      setError(
        e instanceof ApiError && e.status === 429
          ? `You can change it again ${humanDelay(e.data?.retry_after ?? 0)}.`
          : (e as Error)?.message ?? "Could not change your username.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <LimitedField
      label="Username"
      description="How you appear on the leaderboard. Once a week."
      retryAfter={retryAfter}
    >
      {(locked) => (
        <form onSubmit={submit} className="flex gap-2">
          <input
            value={value}
            disabled={locked || busy}
            onChange={(e) => setValue(e.target.value)}
            maxLength={24}
            className={inputClass}
            aria-label="Username"
          />
          <Button
            type="submit"
            variant="secondary"
            className="shrink-0 px-4 py-2"
            disabled={locked || busy || value.trim() === me.username}
          >
            {busy ? "Saving…" : "Save"}
          </Button>
        </form>
      )}
    </LimitedField>
  );
}

function EmailField({
  me,
  retryAfter,
  onChanged,
}: {
  me: Me;
  retryAfter: number;
  onChanged: (retry: number) => void;
}) {
  const [value, setValue] = useState(me.email);
  const [busy, setBusy] = useState(false);
  const [ok, setOk] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setOk(null);
    setError(null);
    setBusy(true);
    try {
      const res = await changeEmail(value.trim());
      onChanged(res.retry_after);
      setOk(res.message);
    } catch (e) {
      setError(
        e instanceof ApiError && e.status === 429
          ? `You can change it again ${humanDelay(e.data?.retry_after ?? 0)}.`
          : (e as Error)?.message ?? "Could not change your email.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <LimitedField
      label="Email"
      description="Used to sign in. Once a day, and the new address has to be confirmed before it takes over."
      retryAfter={retryAfter}
    >
      {(locked) => (
        <>
          <form onSubmit={submit} className="flex gap-2">
            <input
              type="email"
              value={value}
              disabled={locked || busy}
              onChange={(e) => setValue(e.target.value)}
              className={inputClass}
              aria-label="Email address"
            />
            <Button
              type="submit"
              variant="secondary"
              className="shrink-0 px-4 py-2"
              disabled={locked || busy || value.trim() === me.email}
            >
              {busy ? "Sending…" : "Change"}
            </Button>
          </form>
          <Feedback ok={ok} error={error} />
        </>
      )}
    </LimitedField>
  );
}

function PasswordField({
  retryAfter,
  onChanged,
}: {
  retryAfter: number;
  onChanged: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [busy, setBusy] = useState(false);
  const [ok, setOk] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setOk(null);
    setError(null);
    setBusy(true);
    try {
      await changePassword(current, next);
      onChanged();
      setOk("Password changed.");
      setCurrent("");
      setNext("");
      setOpen(false);
    } catch (e) {
      setError(
        e instanceof ApiError && e.status === 429
          ? `You can change it again ${humanDelay(e.data?.retry_after ?? 0)}.`
          : (e as Error)?.message ?? "Could not change your password.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <LimitedField
      label="Password"
      description="Once a day."
      retryAfter={retryAfter}
    >
      {(locked) =>
        !open ? (
          <>
            <Button
              variant="secondary"
              className="px-4 py-2"
              disabled={locked}
              onClick={() => setOpen(true)}
            >
              Change password
            </Button>
            <Feedback ok={ok} error={error} />
          </>
        ) : (
          <form onSubmit={submit} className="flex flex-col gap-2">
            <PasswordInput
              autoComplete="current-password"
              placeholder="Current password"
              required
              value={current}
              onChange={(e) => setCurrent(e.target.value)}
              aria-label="Current password"
            />
            <PasswordInput
              autoComplete="new-password"
              placeholder="New password — at least 8 characters"
              required
              minLength={8}
              value={next}
              onChange={(e) => setNext(e.target.value)}
              aria-label="New password"
            />
            <div className="flex gap-2">
              <Button type="submit" variant="secondary" className="px-4 py-2" disabled={busy}>
                {busy ? "Saving…" : "Save password"}
              </Button>
              <Button
                type="button"
                variant="ghost"
                className="px-4 py-2"
                onClick={() => {
                  setOpen(false);
                  setError(null);
                }}
              >
                Cancel
              </Button>
            </div>
            <Feedback error={error} />
          </form>
        )
      }
    </LimitedField>
  );
}

// ---------------------------------------------------------------------------

function ProfileSection({ me, setMe }: { me: Me; setMe: (m: Me) => void }) {
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function setCountry(code: string) {
    setError(null);
    setSaving(true);
    const previous = me.country;
    setMe({ ...me, country: code || null });   // optimistic
    try {
      const updated = await patchMe({ country: code });
      setMe(updated);
    } catch (e) {
      setMe({ ...me, country: previous });
      setError((e as Error)?.message ?? "Could not save your country.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <SettingsSection
      title="Profile"
      description="What other players see. Every CubeArena profile is public."
    >
      <SettingsRow
        label="Country"
        description="Rep whichever country you want — this is never set from where you happen to be."
        control={
          <CountryPicker
            value={me.country}
            onChange={setCountry}
            disabled={saving}
          />
        }
      >
        <Feedback error={error} />
      </SettingsRow>
    </SettingsSection>
  );
}

function NotificationsSection({ me, setMe }: { me: Me; setMe: (m: Me) => void }) {
  const [error, setError] = useState<string | null>(null);

  async function toggle(field: "notify_invites" | "notify_recap" | "notify_pb", value: boolean) {
    setError(null);
    const previous = me[field];
    setMe({ ...me, [field]: value });
    try {
      setMe(await patchMe({ [field]: value }));
    } catch (e) {
      setMe({ ...me, [field]: previous });
      setError((e as Error)?.message ?? "Could not save that.");
    }
  }

  return (
    <SettingsSection
      title="Notifications"
      description="These now save to your account — but nothing sends yet. There is no scheduler and no live-match feature behind them."
      tag={<ComingSoonTag />}
    >
      <SettingsRow
        label="Match invites"
        description="When someone challenges you to a live match."
        control={
          <Switch
            checked={me.notify_invites}
            onChange={(v) => toggle("notify_invites", v)}
            label="Match invites"
          />
        }
      />
      <SettingsRow
        label="Weekly recap email"
        description="A summary of your solves and rating change."
        control={
          <Switch
            checked={me.notify_recap}
            onChange={(v) => toggle("notify_recap", v)}
            label="Weekly recap email"
          />
        }
      />
      <SettingsRow
        label="New personal best alerts"
        control={
          <Switch
            checked={me.notify_pb}
            onChange={(v) => toggle("notify_pb", v)}
            label="New personal best alerts"
          />
        }
      >
        <Feedback error={error} />
      </SettingsRow>
    </SettingsSection>
  );
}
