import Link from "next/link";
import { Swords, Timer } from "lucide-react";
import { Card } from "@/components/ui/Card";

const modes = [
  {
    href: "/compete/play?mode=solo",
    icon: Timer,
    title: "Solo Practice",
    description: "Scan a scramble, solve it, and verify the result — just you and the clock.",
  },
  {
    href: "/compete/play?mode=live",
    icon: Swords,
    title: "Live Match",
    description: "Race a live opponent on the same scramble. Rated — wins and losses move your ELO.",
  },
];

export default function ChooseModePage() {
  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-16 text-center md:px-10">
      <h1 className="text-5xl font-bold tracking-tight md:text-6xl">Choose a mode</h1>
      <p className="mt-3 text-lg text-ink-faint">Pick how you want to play.</p>

      <div className="mt-12 grid grid-cols-1 gap-6 text-left md:grid-cols-2">
        {modes.map(({ href, icon: Icon, title, description }) => (
          <Link key={href} href={href} className="group">
            <Card className="flex h-72 flex-col overflow-hidden p-0 transition-shadow group-hover:shadow-lg">
              <div className="flex flex-1 items-center justify-center bg-cloud">
                <Icon size={56} strokeWidth={1.25} className="text-ink-faint" />
              </div>
              <div className="border-t border-mist p-6">
                <p className="text-2xl font-medium">{title}</p>
                <p className="mt-1 text-sm text-ink-faint">{description}</p>
              </div>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
