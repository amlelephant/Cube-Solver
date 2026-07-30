import { CompeteFlow } from "@/components/CompeteFlow";

export default async function CompetePlayPage({
  searchParams,
}: {
  searchParams: Promise<{ mode?: string }>;
}) {
  const { mode } = await searchParams;
  return <CompeteFlow mode={mode === "live" ? "live" : "solo"} />;
}
