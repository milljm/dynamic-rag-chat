import { useEffect, useState } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { Check, Lightbulb, Monitor, Moon, Sun } from "lucide-react";
import {
  applyTheme,
  persistThemePref,
  readThemePref,
  resolveTheme,
  type ThemePref,
} from "@/lib/theme";
import { applySyntax, readSyntaxPref } from "@/lib/chat/syntax";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const OPTIONS: {
  value: ThemePref;
  label: string;
  icon: typeof Sun;
}[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
];

export function ThemeToggle() {
  const [pref, setPref] = useState<ThemePref>("dark");

  useEffect(() => {
    const stored = readThemePref();
    setPref(stored);
    const resolved = applyTheme(stored);
    applySyntax(readSyntaxPref(), resolved);
  }, []);

  useEffect(() => {
    if (pref !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const onChange = () => {
      const resolved = applyTheme("system");
      applySyntax(readSyntaxPref(), resolved);
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [pref]);

  function choose(next: ThemePref) {
    setPref(next);
    persistThemePref(next);
    const resolved = applyTheme(next);
    applySyntax(readSyntaxPref(), resolved);
  }

  const resolved = resolveTheme(pref);

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label={`Theme: ${pref}`}
          title={`Theme: ${pref}`}
        >
          <Lightbulb
            className={cn(
              "size-4",
              resolved === "light" && "fill-foreground/20",
            )}
          />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={6}
          className="z-50 min-w-36 rounded-md bg-popover p-1 text-popover-foreground shadow-[var(--shadow-border)]"
        >
          {OPTIONS.map((opt) => {
            const Icon = opt.icon;
            const active = pref === opt.value;
            return (
              <DropdownMenu.Item
                key={opt.value}
                onSelect={() => choose(opt.value)}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-xs outline-none",
                  "data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground",
                )}
              >
                <Icon className="size-3.5 text-muted-foreground" />
                <span className="flex-1">{opt.label}</span>
                {active ? (
                  <Check className="size-3.5 text-foreground" />
                ) : (
                  <span className="size-3.5" />
                )}
              </DropdownMenu.Item>
            );
          })}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
