import Link from "next/link";
import { Github, Instagram, Twitch, Youtube } from "lucide-react";
import { footerColumns } from "@/lib/nav";

const socials = [
  { icon: Github, href: "https://github.com", label: "GitHub" },
  { icon: Twitch, href: "https://twitch.tv", label: "Twitch" },
  { icon: Youtube, href: "https://youtube.com", label: "YouTube" },
  { icon: Instagram, href: "https://instagram.com", label: "Instagram" },
];

export function Footer() {
  return (
    <footer className="mt-24 border-t border-mist">
      <div className="mx-auto max-w-6xl px-6 py-12 md:px-10">
        <div className="flex flex-col justify-between gap-12 md:flex-row">
          <div>
            <p className="text-2xl font-semibold tracking-tight">Cube Arena</p>
            <p className="mt-2 max-w-xs text-sm text-ink-faint">
              Verified speedcubing, straight from your webcam.
            </p>
            <div className="mt-6 flex gap-2">
              {socials.map(({ icon: Icon, href, label }) => (
                <a
                  key={label}
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={label}
                  className="flex size-10 items-center justify-center rounded-md text-ink-soft transition-colors hover:bg-cloud hover:text-ink"
                >
                  <Icon size={20} strokeWidth={1.75} />
                </a>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-8 sm:gap-16">
            {footerColumns.map((col) => (
              <div key={col.topic} className="flex flex-col gap-3">
                <p className="text-sm font-medium text-ink">{col.topic}</p>
                {col.links.map((link, i) => (
                  <Link
                    key={link.label + i}
                    href={link.href}
                    className="text-sm text-ink-faint transition-colors hover:text-ink"
                  >
                    {link.label}
                  </Link>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
