import { useEffect, useState } from "react";
import { Settings2, X } from "lucide-react";
import { toast } from "sonner";
import {
  fetchSettings,
  pingSettings,
  saveSettings,
  ROUTE_ROWS,
  type SettingsKey,
  type SettingsValues,
} from "@/lib/chat/settings";
import { usesChatPy } from "@/lib/chat/remote";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="grid gap-1">
      <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      {children}
      {hint ? (
        <span className="text-[11px] text-muted-foreground">{hint}</span>
      ) : null}
    </label>
  );
}

export function SettingsButton({ streaming }: { streaming: boolean }) {
  const [open, setOpen] = useState(false);
  if (!usesChatPy()) return null;
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        aria-label="Settings"
        title="Settings"
        onClick={() => setOpen(true)}
      >
        <Settings2 className="size-4" />
      </Button>
      {open ? (
        <SettingsPanel streaming={streaming} onClose={() => setOpen(false)} />
      ) : null}
    </>
  );
}

function SettingsPanel({
  streaming,
  onClose,
}: {
  streaming: boolean;
  onClose: () => void;
}) {
  const [values, setValues] = useState<SettingsValues | null>(null);
  const [effective, setEffective] = useState<SettingsValues | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [pingNote, setPingNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [pinging, setPinging] = useState(false);

  function patch(key: SettingsKey, value: string) {
    setValues((prev) => (prev ? { ...prev, [key]: value } : prev));
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const payload = await fetchSettings();
        if (cancelled) return;
        const merged = { ...payload.effective, ...payload.values };
        // Core fields: yaml, then whatever is running. Routes stay yaml-empty = inherit.
        merged.llm_server = payload.values.llm_server || payload.effective.llm_server;
        merged.api_key = payload.values.api_key || payload.effective.api_key || "none";
        merged.model = payload.values.model || payload.effective.model;
        merged.pre_llm = payload.values.pre_llm || payload.effective.pre_llm;
        merged.embedding_llm =
          payload.values.embedding_llm || payload.effective.embedding_llm;
        setValues(merged);
        setEffective(payload.effective);
        if (merged.llm_server) {
          const ping = await pingSettings(merged.llm_server, merged.api_key);
          if (!cancelled && ping.ok) {
            setModels(ping.models);
            setPingNote(`${ping.models.length} models`);
          } else if (!cancelled) {
            setPingNote(ping.error || "unreachable");
          }
        }
      } catch (err) {
        if (!cancelled) toast.error(String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function onPing() {
    if (!values) return;
    setPinging(true);
    try {
      const ping = await pingSettings(values.llm_server, values.api_key);
      if (ping.ok) {
        setModels(ping.models);
        setPingNote(`${ping.models.length} models`);
        toast.success(`Reachable — ${ping.models.length} models`);
      } else {
        setPingNote(ping.error || "unreachable");
        toast.error(ping.error || "Server did not answer");
      }
    } finally {
      setPinging(false);
    }
  }

  async function onSave() {
    if (!values) return;
    if (streaming) {
      toast.error("Wait for the current turn to finish.");
      return;
    }
    setSaving(true);
    try {
      const result = await saveSettings(values);
      if (!result.ok) {
        toast.error(result.error || "Save failed");
        return;
      }
      setValues({ ...result.effective, ...result.values });
      setEffective(result.effective);
      toast.success(result.message || "Saved. Next turn uses these models.");
      onClose();
    } catch (err) {
      toast.error(String(err));
    } finally {
      setSaving(false);
    }
  }

  const listId = "spur-model-list";

  return (
    <div className="fixed inset-0 z-[60] flex justify-end">
      <button
        type="button"
        className="absolute inset-0 bg-background/50"
        aria-label="Close settings"
        onClick={onClose}
      />
      <aside
        role="dialog"
        aria-labelledby="spur-settings-title"
        className="relative flex h-full w-full max-w-md flex-col border-l border-border bg-card paper shadow-[var(--shadow-border)]"
      >
        <header className="flex items-center gap-2 border-b border-border px-4 py-3">
          <h2 id="spur-settings-title" className="flex-1 text-sm font-medium">
            Settings
          </h2>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label="Close"
            onClick={onClose}
          >
            <X className="size-4" />
          </Button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          {!values ? (
            <p className="text-xs text-muted-foreground">Loading…</p>
          ) : (
            <div className="grid gap-5">
              <section className="grid gap-3">
                <h3 className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Server
                </h3>
                <Field label="URL" hint="OpenAI-compatible (LM Studio / Ollama)">
                  <Input
                    value={values.llm_server}
                    onChange={(e) => patch("llm_server", e.target.value)}
                    placeholder="http://127.0.0.1:1234/v1"
                    autoComplete="off"
                    spellCheck={false}
                  />
                </Field>
                <Field label="API key">
                  <Input
                    type="password"
                    value={values.api_key}
                    onChange={(e) => patch("api_key", e.target.value)}
                    autoComplete="off"
                  />
                </Field>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    disabled={pinging || !values.llm_server}
                    onClick={() => void onPing()}
                  >
                    {pinging ? "Pinging…" : "Ping"}
                  </Button>
                  <span className="text-[11px] text-muted-foreground">
                    {pingNote}
                  </span>
                </div>
              </section>

              <section className="grid gap-3">
                <h3 className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Required models
                </h3>
                <Field label="Generator">
                  <Input
                    list={listId}
                    value={values.model}
                    onChange={(e) => patch("model", e.target.value)}
                    spellCheck={false}
                  />
                </Field>
                <Field label="Pre-conditioner">
                  <Input
                    list={listId}
                    value={values.pre_llm}
                    onChange={(e) => patch("pre_llm", e.target.value)}
                    spellCheck={false}
                  />
                </Field>
                <Field label="Embeddings">
                  <Input
                    list={listId}
                    value={values.embedding_llm}
                    onChange={(e) => patch("embedding_llm", e.target.value)}
                    spellCheck={false}
                  />
                </Field>
              </section>

              <details className="border-t border-border pt-3">
                <summary className="cursor-pointer text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Routes
                </summary>
                <p className="mt-2 text-[11px] text-muted-foreground">
                  Blank inherits the generator. Vision and agent stay unset unless
                  you name them.
                </p>
                <div className="mt-3 grid gap-4">
                  {ROUTE_ROWS.map((row) => (
                    <div key={row.id} className="grid gap-1.5">
                      <span className="text-[11px] font-medium text-foreground">
                        {row.label}
                      </span>
                      <Input
                        list={listId}
                        value={values[row.llm]}
                        onChange={(e) => patch(row.llm, e.target.value)}
                        placeholder={
                          row.id === "vision" || row.id === "agent"
                            ? "unset"
                            : effective?.model
                              ? `inherits ${effective.model}`
                              : "inherits generator"
                        }
                        spellCheck={false}
                      />
                      <Input
                        value={values[row.server]}
                        onChange={(e) => patch(row.server, e.target.value)}
                        placeholder="same server"
                        spellCheck={false}
                      />
                    </div>
                  ))}
                </div>
              </details>
              <datalist id={listId}>
                {models.map((name) => (
                  <option key={name} value={name} />
                ))}
              </datalist>
            </div>
          )}
        </div>

        <footer className="border-t border-border px-4 py-3">
          <Button
            type="button"
            className="w-full"
            disabled={!values || saving || streaming}
            onClick={() => void onSave()}
          >
            {saving ? "Saving…" : "Save"}
          </Button>
          <p className="mt-2 text-[11px] text-muted-foreground">
            Writes <span className="font-mono">.chat.yaml</span>. Next turn uses
            these models.
          </p>
        </footer>
      </aside>
    </div>
  );
}
