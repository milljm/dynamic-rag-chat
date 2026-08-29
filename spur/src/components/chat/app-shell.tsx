import { useEffect, useState } from "react";
import { Menu, PanelLeft, X } from "lucide-react";
import { Toaster } from "sonner";
import { useChatStore } from "@/lib/chat/store";
import { usesChatPy } from "@/lib/chat/remote";
import { useSend } from "@/lib/chat/use-send";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Composer } from "./composer";
import { Sidebar } from "./sidebar";
import { Thread } from "./thread";

const SIDEBAR_MIN = 240;
const SIDEBAR_MAX = 560;
const SIDEBAR_DEFAULT = 320;

export function AppShell() {
  const ready = useHydrateChat();
  const { send, stop, regenerate, streaming } = useSend();
  const [navOpen, setNavOpen] = useState(false);
  const sidebar = useSidebarLayout();

  if (!ready) {
    return (
      <div className="flex h-dvh items-center justify-center bg-background text-muted-foreground">
        <p className="font-display text-2xl italic">Spur</p>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <div
        className={cn(
          "relative flex h-dvh overflow-hidden bg-background paper text-foreground",
          sidebar.dragging && "select-none",
        )}
      >
        <div
          className={cn(
            "relative hidden shrink-0 md:block",
            !sidebar.dragging &&
              "transition-[width] duration-[var(--motion-fast)] ease-[cubic-bezier(0.22,1,0.36,1)]",
            sidebar.open ? "border-r border-border" : "border-r-0",
          )}
          style={{
            width: sidebar.open ? sidebar.width : 0,
            overflow: "hidden",
          }}
        >
          <div className="h-full" style={{ width: sidebar.width }}>
            <Sidebar streaming={streaming} onCollapse={() => sidebar.setOpen(false)} />
          </div>
          {sidebar.open && (
            <div
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize sidebar"
              tabIndex={0}
              className="absolute right-0 top-0 z-20 h-full w-1.5 cursor-col-resize hover:bg-foreground/20"
              onPointerDown={(e) => {
                e.preventDefault();
                sidebar.setDragging(true);
                const startX = e.clientX;
                const startW = sidebar.width;
                const move = (ev: PointerEvent) => {
                  const next = Math.min(
                    SIDEBAR_MAX,
                    Math.max(SIDEBAR_MIN, startW + ev.clientX - startX),
                  );
                  sidebar.setWidth(next);
                };
                const up = () => {
                  sidebar.setDragging(false);
                  window.removeEventListener("pointermove", move);
                  window.removeEventListener("pointerup", up);
                };
                window.addEventListener("pointermove", move);
                window.addEventListener("pointerup", up);
              }}
              onKeyDown={(e) => {
                if (e.key === "ArrowLeft") {
                  sidebar.setWidth((w) => Math.max(SIDEBAR_MIN, w - 16));
                }
                if (e.key === "ArrowRight") {
                  sidebar.setWidth((w) => Math.min(SIDEBAR_MAX, w + 16));
                }
              }}
            />
          )}
        </div>

        {navOpen && (
          <div className="fixed inset-0 z-40 md:hidden">
            <button
              type="button"
              className="absolute inset-0 bg-background/70"
              aria-label="Close branches"
              onClick={() => setNavOpen(false)}
            />
            <div className="relative h-full w-[min(20rem,88vw)] border-r border-border bg-card paper shadow-[var(--shadow-border)]">
              <div className="absolute right-2 top-2 z-10">
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Close"
                  onClick={() => setNavOpen(false)}
                >
                  <X />
                </Button>
              </div>
              <Sidebar streaming={streaming} onNavigate={() => setNavOpen(false)} />
            </div>
          </div>
        )}

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex items-center gap-2 border-b border-border px-2 py-1 md:hidden">
            <Button
              variant="ghost"
              size="icon"
              aria-label="Open branches"
              onClick={() => setNavOpen(true)}
            >
              <Menu />
            </Button>
            <span className="font-display text-lg italic">Spur</span>
          </div>
          <Thread
            streaming={streaming}
            onRevealSidebar={
              sidebar.open ? undefined : () => sidebar.setOpen(true)
            }
          />
          <Composer
            streaming={streaming}
            onSend={send}
            onStop={stop}
            onRegenerate={regenerate}
          />
        </div>
      </div>
      <ThemeToaster />
    </TooltipProvider>
  );
}

function ThemeToaster() {
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    typeof document === "undefined"
      ? "dark"
      : document.documentElement.dataset.theme === "light"
        ? "light"
        : "dark",
  );
  useEffect(() => {
    const root = document.documentElement;
    const read = () =>
      setTheme(root.dataset.theme === "light" ? "light" : "dark");
    read();
    const obs = new MutationObserver(read);
    obs.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);
  return (
    <Toaster
      theme={theme}
      position="top-center"
      toastOptions={{
        className:
          "bg-popover text-popover-foreground shadow-[var(--shadow-border)] border-0",
      }}
    />
  );
}

function readStoredWidth(): number {
  if (typeof window === "undefined") return SIDEBAR_DEFAULT;
  try {
    const n = Number(window.localStorage.getItem("spur-sidebar-w"));
    if (Number.isFinite(n)) {
      return Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, n));
    }
  } catch {
    /* ignore */
  }
  return SIDEBAR_DEFAULT;
}

function useSidebarLayout() {
  const [open, setOpen] = useState(() => {
    if (typeof window === "undefined") return true;
    try {
      return window.localStorage.getItem("spur-sidebar") !== "0";
    } catch {
      return true;
    }
  });
  const [width, setWidth] = useState(readStoredWidth);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    try {
      window.localStorage.setItem("spur-sidebar", open ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [open]);

  useEffect(() => {
    try {
      window.localStorage.setItem("spur-sidebar-w", String(Math.round(width)));
    } catch {
      /* ignore */
    }
  }, [width]);

  return { open, setOpen, width, setWidth, dragging, setDragging };
}

function useHydrateChat() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const finish = () => {
      if (!cancelled) setReady(true);
    };

    const boot = async () => {
      if (usesChatPy()) {
        for (let i = 0; i < 8; i++) {
          const ok = await useChatStore.getState().hydrateFromServer();
          if (ok || cancelled) break;
          await new Promise((r) => setTimeout(r, 400 * (i + 1)));
        }
        finish();
        return;
      }
      const api = useChatStore.persist;
      if (!api || typeof api.rehydrate !== "function") {
        finish();
        return;
      }
      try {
        if (api.hasHydrated?.()) {
          finish();
          return;
        }
        await Promise.resolve(api.rehydrate());
      } catch {
        /* show seed */
      }
      finish();
    };

    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  return ready;
}