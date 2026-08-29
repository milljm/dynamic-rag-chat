import type { ReactNode } from "react";
import { BookOpen, Bot, Lock } from "lucide-react";
import { isLockedBranch, modeOf } from "@/lib/chat/branch-mode";
import type { Mode } from "@/lib/chat/types";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export function ModeToggle({
  branchId,
  mode,
  onChange,
}: {
  branchId: string;
  mode: Mode;
  onChange: (mode: Mode) => void;
}) {
  const locked = isLockedBranch(branchId);
  const effective = modeOf({ id: branchId, mode });

  return (
    <div className="space-y-2">
      <div
        role="radiogroup"
        aria-label="Conversation mode"
        aria-disabled={locked}
        className="grid grid-cols-2 gap-1 rounded-md bg-secondary p-1"
      >
        <ModeOption
          value="assistant"
          active={effective === "assistant"}
          locked={locked}
          onSelect={onChange}
          icon={<Bot className="size-3.5" />}
          label="Assistant"
        />
        <ModeOption
          value="story"
          active={effective === "story"}
          locked={locked}
          onSelect={onChange}
          icon={<BookOpen className="size-3.5" />}
          label="Story"
        />
      </div>
      <p className="flex items-start gap-1.5 text-xs leading-snug text-muted-foreground">
        {locked ? (
          <>
            <Lock className="mt-0.5 size-3 shrink-0" />
            Protected branch — mode is fixed. Fork a branch to switch flavors.
          </>
        ) : (
          "Switch system prompt flavor. This only changes the current branch."
        )}
      </p>
    </div>
  );
}

function ModeOption({
  value,
  active,
  locked,
  onSelect,
  icon,
  label,
}: {
  value: Mode;
  active: boolean;
  locked: boolean;
  onSelect: (mode: Mode) => void;
  icon: ReactNode;
  label: string;
}) {
  const disabled = locked && !active;
  const button = (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      disabled={disabled}
      onClick={() => {
            if (active || (locked && !active)) return;
            onSelect(value);
          }}
      className={cn(
        "relative z-10 flex h-9 items-center justify-center gap-1.5 rounded-sm text-xs font-medium transition-colors duration-150",
        active ? "bg-primary text-primary-foreground" : "text-muted-foreground",
        disabled && "opacity-40",
      )}
    >
      {icon}
      {label}
    </button>
  );

  if (!disabled) return button;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="grid">{button}</span>
      </TooltipTrigger>
      <TooltipContent>
        Create a new branch to use {label.toLowerCase()} mode.
      </TooltipContent>
    </Tooltip>
  );
}
