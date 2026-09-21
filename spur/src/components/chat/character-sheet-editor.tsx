import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  fetchCharacterSheet,
  saveCharacterSheet,
} from "@/lib/chat/character-sheet";
import { usesChatPy } from "@/lib/chat/remote";
import { Button } from "@/components/ui/button";

export function CharacterSheetEditor({ onClose }: { onClose: () => void }) {
  const [content, setContent] = useState("");
  const [path, setPath] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const savedRef = useRef("");

  useEffect(() => {
    if (!usesChatPy()) {
      toast.error("The character sheet is served by ./chat.py --spur.");
      onClose();
      return;
    }
    let cancelled = false;
    setLoading(true);
    void fetchCharacterSheet().then((sheet) => {
      if (cancelled) return;
      setLoading(false);
      if (!sheet.ok) {
        toast.error(sheet.error || "Could not load the character sheet.");
        if (sheet.enabled === false) onClose();
        return;
      }
      setContent(sheet.content ?? "");
      savedRef.current = sheet.content ?? "";
      setPath(sheet.path || "");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const dirty = content !== savedRef.current;

  async function save() {
    setSaving(true);
    const sheet = await saveCharacterSheet(content);
    setSaving(false);
    if (!sheet.ok) {
      toast.error(sheet.error || "Save failed.");
      return;
    }
    savedRef.current = content;
    if (sheet.path) setPath(sheet.path);
    toast.success(sheet.message || "Saved.");
  }

  function close() {
    if (dirty && !window.confirm("Discard unsaved character sheet changes?")) {
      return;
    }
    onClose();
  }

  const title = "Character sheet";

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex items-center gap-3 border-b border-border px-4 py-3 md:px-8">
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-sm font-medium">{title}</h1>
          <p className="truncate font-mono text-[11px] text-muted-foreground">
            {path || "character_sheet"}
          </p>
        </div>
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
          <p className="text-sm text-muted-foreground">
            Loading character sheet…
          </p>
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
