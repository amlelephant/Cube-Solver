"use client";

import { useState, type ReactNode } from "react";
import { Sun, Moon } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { Switch } from "@/components/ui/Switch";
import { useTheme } from "@/lib/theme";

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
}: {
  label: string;
  description?: string;
  control: ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-6 border-t border-mist pt-5 first:border-t-0 first:pt-0">
      <div>
        <p className="text-sm font-medium">{label}</p>
        {description && <p className="mt-0.5 text-xs text-ink-faint">{description}</p>}
      </div>
      {control}
    </div>
  );
}

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();

  const [notifyInvites, setNotifyInvites] = useState(true);
  const [notifyRecap, setNotifyRecap] = useState(false);
  const [notifyPb, setNotifyPb] = useState(true);
  const [publicProfile, setPublicProfile] = useState(true);
  const [showCountry, setShowCountry] = useState(true);

  return (
    <div className="mx-auto max-w-3xl px-6 pb-24 pt-16 md:px-10">
      <h1 className="text-5xl font-bold tracking-tight md:text-6xl">Settings</h1>
      <p className="mt-3 text-lg text-ink-faint">
        Most of this is a preview for now — appearance is live, the rest is coming soon.
        Looking for avatar cosmetics? Click your avatar on your profile.
      </p>

      <div className="mt-10 flex flex-col gap-6">
        <SettingsSection title="Appearance" description="Choose how CubeArena looks on this device.">
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

        <SettingsSection
          title="Account"
          description="Basic profile details."
          tag={<ComingSoonTag />}
        >
          <SettingsRow
            label="Username"
            control={
              <input
                disabled
                value="Aiden"
                className="w-40 rounded-lg border border-mist bg-cloud px-3 py-2 text-sm text-ink-faint"
              />
            }
          />
          <SettingsRow
            label="Email"
            control={
              <input
                disabled
                placeholder="you@example.com"
                className="w-48 rounded-lg border border-mist bg-cloud px-3 py-2 text-sm text-ink-faint placeholder:text-ink-faint"
              />
            }
          />
          <SettingsRow
            label="Password"
            control={
              <button
                disabled
                className="cursor-not-allowed rounded-lg bg-cloud px-4 py-2 text-sm font-medium text-ink-faint"
              >
                Change password
              </button>
            }
          />
        </SettingsSection>

        <SettingsSection
          title="Notifications"
          description="These toggle locally for now — nothing is wired up to send anything yet."
          tag={<ComingSoonTag />}
        >
          <SettingsRow
            label="Match invites"
            description="When someone challenges you to a live match."
            control={<Switch checked={notifyInvites} onChange={setNotifyInvites} label="Match invites" />}
          />
          <SettingsRow
            label="Weekly recap email"
            description="A summary of your solves and rating change."
            control={<Switch checked={notifyRecap} onChange={setNotifyRecap} label="Weekly recap email" />}
          />
          <SettingsRow
            label="New personal best alerts"
            control={<Switch checked={notifyPb} onChange={setNotifyPb} label="New personal best alerts" />}
          />
        </SettingsSection>

        <SettingsSection
          title="Privacy"
          description="Control what other players can see."
          tag={<ComingSoonTag />}
        >
          <SettingsRow
            label="Public profile"
            description="Let anyone view your stats and history."
            control={<Switch checked={publicProfile} onChange={setPublicProfile} label="Public profile" />}
          />
          <SettingsRow
            label="Show country flag"
            control={<Switch checked={showCountry} onChange={setShowCountry} label="Show country flag" />}
          />
        </SettingsSection>
      </div>
    </div>
  );
}
