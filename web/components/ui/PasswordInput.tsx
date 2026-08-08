"use client";

import { useId, useState, type InputHTMLAttributes } from "react";
import { Eye, EyeOff } from "lucide-react";
import { cn } from "@/lib/cn";

/**
 * A password field with a show/hide toggle.
 *
 * Three details that are easy to miss and all matter:
 *
 *  * `type="button"` on the toggle. A bare <button> inside a <form> defaults
 *    to submit, so revealing your password would submit the form.
 *  * The toggle is NOT in the tab order (`tabIndex={-1}`). Tabbing out of a
 *    password field should land on the submit button; a reveal control in
 *    between is a trip hazard for keyboard users on every single sign-in.
 *  * Revealing sets `aria-pressed` and swaps the label, so a screen reader
 *    hears the current state rather than an unchanging "show password".
 *
 * The visible state deliberately does not persist anywhere. A field that
 * remembers it was revealed will one day be revealed in front of someone.
 */
export function PasswordInput({
  className,
  ...props
}: Omit<InputHTMLAttributes<HTMLInputElement>, "type">) {
  const [shown, setShown] = useState(false);
  const describedBy = useId();

  return (
    <div className="relative">
      <input
        {...props}
        type={shown ? "text" : "password"}
        className={cn(
          "w-full rounded-lg border border-mist bg-cloud py-2.5 pl-3.5 pr-11 text-sm",
          "text-ink placeholder:text-ink-faint",
          "outline-none transition-colors focus:border-ink/30 focus:bg-paper",
          className,
        )}
      />
      <button
        type="button"
        tabIndex={-1}
        onClick={() => setShown((s) => !s)}
        aria-pressed={shown}
        aria-label={shown ? "Hide password" : "Show password"}
        aria-describedby={describedBy}
        title={shown ? "Hide password" : "Show password"}
        className={cn(
          "absolute right-1 top-1/2 -translate-y-1/2 rounded-md p-2",
          "text-ink-faint transition-colors hover:text-ink",
          "focus-visible:outline focus-visible:outline-2 focus-visible:outline-ink/30",
        )}
      >
        {shown ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
      <span id={describedBy} className="sr-only">
        {shown ? "Password is visible" : "Password is hidden"}
      </span>
    </div>
  );
}
