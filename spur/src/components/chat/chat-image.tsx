import { useEffect, useState, type MouseEvent } from "react";
import { createPortal } from "react-dom";
import { Download, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function ChatImage({
  src,
  alt,
  name,
  className,
}: {
  src: string;
  alt: string;
  name?: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open]);

  function save(e: MouseEvent) {
    e.stopPropagation();
    const a = document.createElement("a");
    a.href = src;
    a.download = name || alt || "image.png";
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="mt-2 block max-w-full cursor-zoom-in rounded-sm text-left"
        title="View full size"
      >
        <img
          src={src}
          alt={alt}
          className={cn(
            "max-h-48 rounded-sm outline outline-1 -outline-offset-1 outline-foreground/10",
            className,
          )}
        />
      </button>
      {open
        ? createPortal(
            <div
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
              onClick={() => setOpen(false)}
              role="dialog"
              aria-modal="true"
              aria-label={alt || "Image"}
            >
              <div className="absolute right-4 top-4 flex gap-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={save}
                  title="Save image"
                >
                  <Download className="size-4" />
                  Save
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  size="icon-sm"
                  onClick={() => setOpen(false)}
                  aria-label="Close"
                >
                  <X className="size-4" />
                </Button>
              </div>
              <img
                src={src}
                alt={alt}
                onClick={(e) => e.stopPropagation()}
                className="max-h-[92vh] max-w-[92vw] object-contain shadow-2xl"
              />
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
