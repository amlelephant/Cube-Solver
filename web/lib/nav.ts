export type NavItem = {
  href: string;
  label: string;
  /**
   * Show a lock beside the label for accounts without Coach. The page is
   * still reachable — it is where you go to buy it — so this is a marker,
   * never a guard. The real gate is server-side in
   * `core/views.solve_analysis`.
   */
  premium?: boolean;
};

export const primaryNav: NavItem[] = [
  { href: "/home", label: "Home" },
  { href: "/compete", label: "Play" },
  { href: "/analytics", label: "Analytics" },
  { href: "/coach", label: "Coach", premium: true },
  { href: "/leaderboard", label: "Leaderboard" },
];

export const footerColumns = [
  {
    topic: "Arena",
    links: [
      { href: "/home", label: "Home" },
      { href: "/compete", label: "Play" },
      { href: "/analytics", label: "Analytics" },
      { href: "/coach", label: "Coach" },
      { href: "/leaderboard", label: "Leaderboard" },
    ],
  },
  {
    topic: "Account",
    links: [
      { href: "/profile", label: "Profile" },
      { href: "/profile", label: "Match history" },
      { href: "/profile", label: "Statistics" },
      { href: "/settings", label: "Settings" },
    ],
  },
  {
    topic: "Project",
    links: [
      { href: "/", label: "About CubeArena" },
      { href: "/", label: "How verification works" },
      { href: "/", label: "Early access" },
    ],
  },
];
