import { type ReactNode, type HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export function Card({
  className,
  children,
  ...props
}: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-mist bg-paper shadow-card",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
