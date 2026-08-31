import { useEffect, useRef, useState } from "react";
import { Settings2, X } from "lucide-react";
import { toast } from "sonner";
import {
  fetchSettings,
  hostForRole,
  normalizeHost,
  pingSettings,
  pingSd,
  saveSettings,
  uniqueRoleHosts,
  ROUTE_GROUPS,
  type ModelInfo,
  type PingResult,
  type SettingsKey,
  type SettingsValues,
} from "@/lib/chat/settings";
import { usesChatPy } from "@/lib/chat/remote";
import { useChatStore } from "@/lib/chat/store";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const fieldClass =
  "flex h-10 w-full rounded-sm bg-secondary px-3 text-sm text-foreground shadow-[var(--shadow-border)] transition-[box-shadow] duration-150 placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/70 disabled:cursor-not-allowed disabled:opacity-50";

const fieldLabelClass =
  "text-[11px] font-medium uppercase tracking-wide text-muted-foreground";

const modelLabelClass =
  "font-display text-[11px] font-semibold uppercase tracking-[0.32em] text-route";

type Catalog = {
  models: string[];
  details: ModelInfo[];
  note: string;
  ok: boolean;
};

const EMPTY_CATALOG: Catalog = {
  models: [],
  details: [],
  note: "",
  ok: false,
};

function catalogFromPing(ping: PingResult): Catalog {
  const rows = (
    ping.details?.length
      ? ping.details
      : ping.models.map((id) => ({ id, loaded: null as boolean | null }))
  )
    .slice()
    .sort((a, b) => {
      if (Boolean(a.loaded) !== Boolean(b.loaded)) return a.loaded ? -1 : 1;
      return a.id.localeCompare(b.id);
    });
  const hot = rows.filter((row) => row.loaded).length;
  return {
    models: rows.map((row) => row.id),
    details: rows,
    note: ping.knows_loaded
      ? `${hot} loaded · ${rows.length} downloaded`
      : `${rows.length} models`,
    ok: true,
  };
}

function failedCatalog(error: string): Catalog {
  return { models: [], details: [], note: error, ok: false };
}

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
      <span className={fieldLabelClass}>
        {label}
      </span>
      {children}
      {hint ? (
        <span className="text-[11px] text-muted-foreground">{hint}</span>
      ) : null}
    </label>
  );
}

function ModelSelect({
  value,
  onChange,
  models,
  details,
  emptyLabel,
  required,
}: {
  value: string;
  onChange: (next: string) => void;
  models: string[];
  details: ModelInfo[];
  emptyLabel?: string;
  required?: boolean;
}) {
  const info: ModelInfo[] = details.length
    ? details
    : models.map((id) => ({ id, loaded: null }));
  const ids = info.map((row) => row.id);
  if (value && !ids.includes(value)) {
    info.unshift({ id: value, loaded: null });
  }
  if (info.length === 0) {
    return (
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={emptyLabel}
        spellCheck={false}
      />
    );
  }
  const knows = info.some((row) => row.loaded === true || row.loaded === false);
  const hot = info.filter((row) => row.loaded);
  const rest = info.filter((row) => !row.loaded);
  const renderOption = (row: ModelInfo) => (
    <option key={row.id} value={row.id}>
      {row.loaded ? `● ${row.id}` : row.id}
    </option>
  );
  return (
    <select
      className={cn(fieldClass, "appearance-auto")}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {emptyLabel ? <option value="">{emptyLabel}</option> : null}
      {required && !value ? (
        <option value="" disabled>
          Select a model…
        </option>
      ) : null}
      {knows ? (
        <>
          {hot.length ? (
            <optgroup label="Loaded">{hot.map(renderOption)}</optgroup>
          ) : null}
          {rest.length ? (
            <optgroup label="Downloaded">{rest.map(renderOption)}</optgroup>
          ) : null}
        </>
      ) : (
        info.map(renderOption)
      )}
    </select>
  );
}

export function SettingsButton({ streaming }: { streaming: boolean }) {
  const needsSetup = useChatStore((s) => s.needsSetup);
  const [open, setOpen] = useState(false);
  const openedOnce = useRef(false);
  useEffect(() => {
    if (needsSetup && !openedOnce.current) {
      openedOnce.current = true;
      setOpen(true);
    }
  }, [needsSetup]);
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
  const [catalogs, setCatalogs] = useState<Record<string, Catalog>>({});
  const [pingNote, setPingNote] = useState("");
  const [sdNote, setSdNote] = useState("");
  const [sdModels, setSdModels] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [pinging, setPinging] = useState(false);
  const [sdPinging, setSdPinging] = useState(false);

  function patch(key: SettingsKey, value: string) {
    setValues((prev) => (prev ? { ...prev, [key]: value } : prev));
  }

  function catalogFor(roleServer: string): Catalog {
    const host = hostForRole(roleServer, values?.llm_server || "");
    return (host && catalogs[host]) || EMPTY_CATALOG;
  }

  function storeCatalog(host: string, catalog: Catalog) {
    const key = normalizeHost(host);
    if (!key) return;
    setCatalogs((prev) => ({ ...prev, [key]: catalog }));
    if (key === normalizeHost(values?.llm_server || host)) {
      setPingNote(catalog.note);
    }
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
        merged.llm_server = payload.values.llm_server || payload.effective.llm_server;
        merged.api_key = payload.values.api_key || payload.effective.api_key || "none";
        merged.model = payload.values.model || payload.effective.model;
        merged.pre_llm = payload.values.pre_llm || payload.effective.pre_llm;
        merged.embedding_llm =
          payload.values.embedding_llm || payload.effective.embedding_llm;
        setValues(merged);
        setEffective(payload.effective);
        const hosts = uniqueRoleHosts(merged);
        const results = await Promise.all(
          hosts.map(async (host) => {
            const ping = await pingSettings(host, merged.api_key);
            return { host, ping };
          }),
        );
        if (cancelled) return;
        const next: Record<string, Catalog> = {};
        for (const { host, ping } of results) {
          next[normalizeHost(host)] = ping.ok
            ? catalogFromPing(ping)
            : failedCatalog(ping.error || "unreachable");
        }
        setCatalogs(next);
        const main = next[normalizeHost(merged.llm_server)];
        if (main) setPingNote(main.note);
        if (merged.sd_server) {
          const sd = await pingSd(merged.sd_server);
          if (cancelled) return;
          if (sd.ok) {
            setSdModels(sd.models);
            setSdNote(
              sd.current
                ? `${sd.models.length} checkpoints · loaded`
                : `${sd.models.length} checkpoints`,
            );
            if (sd.current && !merged.sd_model) {
              setValues((prev) =>
                prev ? { ...prev, sd_model: sd.current || "" } : prev,
              );
            }
          } else {
            setSdNote(sd.error || "unreachable");
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

  async function onPing(
    host = values?.llm_server,
    key = values?.api_key,
    noisy = true,
  ) {
    if (!host) return;
    const isMain = normalizeHost(host) === normalizeHost(values?.llm_server || host);
    if (isMain) setPinging(true);
    try {
      const ping = await pingSettings(host, key || "none");
      if (ping.ok) {
        storeCatalog(host, catalogFromPing(ping));
        if (noisy) {
          const hot = (ping.loaded || []).length;
          toast.success(
            ping.knows_loaded
              ? `Reachable — ${hot} loaded · ${ping.models.length} downloaded`
              : `Reachable — ${ping.models.length} models`,
          );
        }
      } else {
        storeCatalog(host, failedCatalog(ping.error || "unreachable"));
        if (noisy) toast.error(ping.error || "Server did not answer");
      }
    } finally {
      if (isMain) setPinging(false);
    }
  }

  function onRoleServerBlur(host: string) {
    const trimmed = host.trim();
    if (!trimmed) return;
    void onPing(trimmed, values?.api_key, false);
  }

  async function onSdPing() {
    if (!values?.sd_server) return;
    setSdPinging(true);
    try {
      const ping = await pingSd(values.sd_server);
      if (ping.ok) {
        setSdModels(ping.models);
        setSdNote(
          ping.current
            ? `${ping.models.length} checkpoints · loaded`
            : `${ping.models.length} checkpoints`,
        );
        if (ping.current && !values.sd_model) {
          patch("sd_model", ping.current);
        }
        toast.success(`Automatic1111 — ${ping.models.length} checkpoints`);
      } else {
        setSdNote(ping.error || "unreachable");
        toast.error(ping.error || "Stable Diffusion did not answer");
      }
    } finally {
      setSdPinging(false);
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
      await useChatStore.getState().hydrateFromServer();
      onClose();
    } catch (err) {
      toast.error(String(err));
    } finally {
      setSaving(false);
    }
  }

  const mainCat = catalogFor("");
  const preCat = catalogFor(values?.pre_server || "");
  const embedCat = catalogFor(values?.embedding_server || "");

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
                    onBlur={(e) => {
                      if (e.target.value) void onPing(e.target.value, values.api_key, false);
                    }}
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
                <Field label="Generator" hint="Uses the server above">
                  <ModelSelect
                    required
                    value={values.model}
                    onChange={(v) => patch("model", v)}
                    models={mainCat.models}
                    details={mainCat.details}
                  />
                </Field>
                <div className="grid gap-1.5">
                  <span className={fieldLabelClass}>Pre-conditioner</span>
                  <ModelSelect
                    required
                    value={values.pre_llm}
                    onChange={(v) => patch("pre_llm", v)}
                    models={preCat.models}
                    details={preCat.details}
                  />
                  <Input
                    value={values.pre_server}
                    onChange={(e) => patch("pre_server", e.target.value)}
                    onBlur={(e) => onRoleServerBlur(e.target.value)}
                    placeholder="same server"
                    spellCheck={false}
                  />
                  {values.pre_server && preCat.note ? (
                    <span className="text-[11px] text-muted-foreground">
                      {preCat.note}
                    </span>
                  ) : (
                    <span className="text-[11px] text-muted-foreground">
                      Blank server inherits the generator server
                    </span>
                  )}
                </div>
                <div className="grid gap-1.5">
                  <span className={fieldLabelClass}>Embeddings</span>
                  <ModelSelect
                    required
                    value={values.embedding_llm}
                    onChange={(v) => patch("embedding_llm", v)}
                    models={embedCat.models}
                    details={embedCat.details}
                  />
                  <Input
                    value={values.embedding_server}
                    onChange={(e) => patch("embedding_server", e.target.value)}
                    onBlur={(e) => onRoleServerBlur(e.target.value)}
                    placeholder="same server"
                    spellCheck={false}
                  />
                  {values.embedding_server && embedCat.note ? (
                    <span className="text-[11px] text-muted-foreground">
                      {embedCat.note}
                    </span>
                  ) : (
                    <span className="text-[11px] text-muted-foreground">
                      Blank server inherits the generator server
                    </span>
                  )}
                </div>
              </section>

              <details className="border-t border-border pt-3">
                <summary className="cursor-pointer text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Routes
                </summary>
                <p className="mt-2 text-[11px] text-muted-foreground">
                  Blank inherits the generator. Vision, agent, and polisher stay
                  unset unless you pick one. Unset polisher skips the extra
                  generation pass. Each dropdown lists models from that row's
                  server.
                </p>
                <div className="mt-3 grid gap-4">
                  {ROUTE_GROUPS.map((group, gi) => (
                    <div
                      key={group.title}
                      className={
                        gi > 0
                          ? "mt-1 border-t border-border pt-4"
                          : undefined
                      }
                    >
                      <h4 className="mb-3 text-[11px] font-medium text-muted-foreground">
                        {group.title}
                      </h4>
                      <div className="grid gap-4">
                  {group.rows.map((row) => {
                    const cat = catalogFor(values[row.server]);
                    return (
                    <div key={row.id} className="grid gap-1.5">
                      <span className={modelLabelClass}>
                        {row.label}
                      </span>
                      <ModelSelect
                        value={values[row.llm]}
                        onChange={(v) => patch(row.llm, v)}
                        models={cat.models}
                        details={cat.details}
                        emptyLabel={
                          row.id === "vision" ||
                          row.id === "agent" ||
                          row.id === "polisher"
                            ? "unset"
                            : effective?.model
                              ? `inherits ${effective.model}`
                              : "inherits generator"
                        }
                      />
                      <Input
                        value={values[row.server]}
                        onChange={(e) => patch(row.server, e.target.value)}
                        onBlur={(e) => onRoleServerBlur(e.target.value)}
                        placeholder="same server"
                        spellCheck={false}
                      />
                      {values[row.server] && cat.note ? (
                        <span className="text-[11px] text-muted-foreground">
                          {cat.note}
                        </span>
                      ) : null}
                      {row.id === "agent" ? (
                        <Field
                          label="Tavily"
                          hint="Tavily API Key or blank for DuckDuckGo"
                        >
                          <Input
                            type="password"
                            value={values.tavily_key}
                            onChange={(e) => patch("tavily_key", e.target.value)}
                            autoComplete="off"
                          />
                        </Field>
                      ) : null}
                    </div>
                    );
                  })}
                      </div>
                    </div>
                  ))}
                </div>
              </details>
              <details className="border-t border-border pt-3">
                <summary className="cursor-pointer text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Stable Diffusion
                </summary>
                  <p className="mt-2 text-[11px] text-muted-foreground">
                    Needs a model that can call tools (set Agent, or it uses
                    the generator). Vision is not used — the picture lands in
                    the chat. The agent writes the prompt; it does not look
                    at the pixels.
                  </p>
                  <div className="mt-3 grid gap-3">
                    <Field
                      label="Automatic1111"
                      hint="http://host:7860 with --api. Blank disables."
                    >
                      <Input
                        value={values.sd_server}
                        onChange={(e) => patch("sd_server", e.target.value)}
                        placeholder="http://127.0.0.1:7860"
                        autoComplete="off"
                        spellCheck={false}
                      />
                    </Field>
                    <Field
                      label="Checkpoint"
                      hint="Swaps the A1111 model on the next generate. Blank keeps whatever is loaded."
                    >
                      <ModelSelect
                        value={values.sd_model}
                        onChange={(v) => patch("sd_model", v)}
                        models={sdModels}
                        details={[]}
                        emptyLabel="A1111 default"
                      />
                    </Field>
                    <div className="flex items-center gap-2">
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        disabled={sdPinging || !values.sd_server}
                        onClick={() => void onSdPing()}
                      >
                        {sdPinging ? "Pinging…" : "Ping"}
                      </Button>
                      <span className="text-[11px] text-muted-foreground">
                        {sdNote}
                      </span>
                    </div>
                  </div>
                </details>
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
