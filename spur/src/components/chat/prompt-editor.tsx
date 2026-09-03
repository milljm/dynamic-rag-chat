import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { modeOf } from "@/lib/chat/branch-mode";
import {
  fetchPrompt,
  restorePrompt,
  savePrompt,
  usesChatPy,
} from "@/lib/chat/remote";
import { useChatStore } from "@/lib/chat/store";
import { Button } from "@/components/ui/button";

export function PromptEditor({ onClose }: { onClose: () => void }) {
  const branch = useChatStore((s) => s.branches[s.currentId]);
  const mode = branch ? modeOf(branch) : "story";
  const [content, setContent] = useState("");
  const [path, setPath] = useState("");
  const [overlaid, setOverlaid] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const savedRef = useRef("");

  function applySlot(slot: {
    content?: string;
    path?: string;
    overlaid?: boolean;
  }) {
    setContent(slot.content ?? "");
    savedRef.current = slot.content ?? "";
    setPath(slot.path || "");
    setOverlaid(Boolean(slot.overlaid));
  }

  useEffect(() => {
    if (!usesChatPy()) {
      toast.error("Prompt files are served by ./chat.py --spur.");
      onClose();
      return;
    }
    let cancelled = false;
    setLoading(true);
    void fetchPrompt(mode).then((slot) => {
      if (cancelled) return;
      setLoading(false);
      if (!slot.ok) {
        toast.error(slot.error || "Could not load prompt.");
        return;
      }
      applySlot(slot);
    });
    return () => {
      cancelled = true;
    };
  }, [mode]);

  const dirty = content !== savedRef.current;

  async function save() {
    setSaving(true);
    const slot = await savePrompt(mode, content);
    setSaving(false);
    if (!slot.ok) {
      toast.error(slot.error || "Save failed.");
      return;
    }
    savedRef.current = content;
    setOverlaid(true);
    if (slot.path) setPath(slot.path);
    toast.success(slot.message || "Saved.");
  }

  async function restore() {
    if (
      !window.confirm(
        "Restore the original system prompt? Your edited copy will be discarded.",
      )
    ) {
      return;
    }
    setSaving(true);
    const slot = await restorePrompt(mode);
    setSaving(false);
    if (!slot.ok) {
      toast.error(slot.error || "Restore failed.");
      return;
    }
    applySlot(slot);
    toast.success(slot.message || "Restored.");
  }

  function close() {
    if (dirty && !window.confirm("Discard unsaved prompt changes?")) return;
    onClose();
  }

  const title = `${mode === "assistant" ? "Assistant" : "Story"} · system prompt`;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex items-center gap-3 border-b border-border px-4 py-3 md:px-8">
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-sm font-medium">{title}</h1>
          <p className="truncate font-mono text-[11px] text-muted-foreground">
            {path || "prompts/…"}
            {overlaid ? " · overlay" : ""}
          </p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={saving || loading || (!overlaid && !dirty)}
          onClick={() => void restore()}
        >
          Restore original
        </Button>
        <Button
          type="button"
          variant="secondary"
          size="sm"
          disabled={!dirty || saving || loading}
          onClick={() => void save()}
        >
          {saving ? "Saving…" : "Save"}
        </Button>
        <Button type="button" variant="ghost" size="sm" onClick={close}>
          Close
        </Button>
      </header>
      <div className="min-h-0 flex-1 p-3 md:p-4">
        {loading ? (
          <p className="text-sm text-muted-foreground">Loading prompt…</p>
        ) : (
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            spellCheck={false}
            className="h-full min-h-[12rem] w-full resize-none rounded-sm bg-secondary p-3 font-mono text-xs leading-relaxed text-foreground shadow-[var(--shadow-border)] outline-none focus-visible:ring-2 focus-visible:ring-ring/70"
            aria-label={title}
          />
        )}
      </div>
    </div>
  );
}
