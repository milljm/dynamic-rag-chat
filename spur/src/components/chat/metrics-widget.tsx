/**
 * Sidebar "Performance" widget: a line chart of the per-response footer
 * metrics (TTFT / Gen / Tok / T/s / Dup / CTX) across the active branch's
 * turns. Values come straight from each message's `metrics` (same object the
 * bubble footer renders), so the chart and the footer never disagree.
 *
 * When the branch has reasoning responses, the Gen and Tok charts gain a
 * second orange line: the non-reasoning (answer) share of that turn.
 *
 * Adapted from `contrib_examples/frontend/src/components/chat/metrics-sidebar.tsx`,
 * collapsed to a single line (one assistant per turn — no per-speaker split).
 */
import {
  memo,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent,
  type ReactNode,
} from "react";
import { useShallow } from "zustand/react/shallow";
import {
  METRICS,
  altSeries,
  fmtTick,
  hasReasoning,
  historyFromSamples,
  lastValue,
  metricSeries,
  niceMax,
  perfSamples,
  type MetricDef,
  type PerfTurn,
} from "@/lib/chat/perf";
import { useChatStore } from "@/lib/chat/store";
import { cn } from "@/lib/utils";

const CHART_H = 210;
const PAD = { t: 10, r: 12, b: 18, l: 34 };

/** One chart legend / hover-tooltip row: [label, textClass, dotClass, value]. */
type LegendRow = [label: string, textCls: string, dotCls: string, v: number | null];

/** Fallback label for the orange non-reasoning line (metrics may override via `altTooltipLabel`). */
const ALT_LABEL = "Non-reasoning";

const PerfChart = memo(function PerfChart({
  history,
  series,
  alt,
  metric,
}: {
  history: PerfTurn[];
  series: (number | null)[];
  alt: (number | null)[] | null;
  metric: MetricDef;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  /** Hovered turn index (0-based into `history`). */
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setWidth(entry.contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const maxTurn = history.length;
  const innerW = Math.max(0, width - PAD.l - PAD.r);
  const innerH = CHART_H - PAD.t - PAD.b;

  const yMax = useMemo(() => {
    let m = 0;
    const scan = (arr: (number | null)[]) => {
      for (const v of arr) if (v != null && Number.isFinite(v)) m = Math.max(m, v);
    };
    scan(series);
    if (alt) scan(alt);
    return niceMax(m);
  }, [series, alt]);

  const xAt = (i: number) => PAD.l + (maxTurn <= 1 ? innerW / 2 : (i / (maxTurn - 1)) * innerW);
  const yAt = (v: number) => PAD.t + (1 - v / yMax) * innerH;

  const buildPath = (arr: (number | null)[]) => {
    let d = "";
    arr.forEach((v, i) => {
      if (v == null || !Number.isFinite(v)) return;
      d += `${d ? "L" : "M"}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`;
    });
    return d;
  };
  const dotR = maxTurn > 60 ? 1.5 : 2.25;
  const buildDots = (arr: (number | null)[], cls: string): ReactNode[] => {
    const out: ReactNode[] = [];
    arr.forEach((v, i) => {
      if (v == null || !Number.isFinite(v)) return;
      out.push(<circle key={`${cls}-${i}`} cx={xAt(i)} cy={yAt(v)} r={dotR} className={cls} />);
    });
    return out;
  };

  const xTicks = useMemo(() => {
    if (maxTurn === 0) return [];
    if (maxTurn <= 2) return history.map((_, i) => i);
    const mid = Math.floor((maxTurn - 1) / 2);
    return [...new Set([0, mid, maxTurn - 1])].sort((a, b) => a - b);
  }, [maxTurn, history]);

  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    if (maxTurn === 0 || innerW <= 0) return;
    const x = e.clientX - e.currentTarget.getBoundingClientRect().left;
    const frac = (x - PAD.l) / innerW;
    const idx = Math.round(frac * (maxTurn - 1));
    setHover(Math.min(maxTurn - 1, Math.max(0, idx)));
  };

  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const hoverX = hover != null ? xAt(hover) : 0;
  const flip = hoverX > width / 2;

  // Measured (pre-paint) tooltip width, so the single-line bubble can be kept
  // fully inside the chart — the sidebar clips horizontal overflow.
  const tipRef = useRef<HTMLDivElement | null>(null);
  const [tipW, setTipW] = useState(0);
  useLayoutEffect(() => {
    if (hover == null) return;
    const w = tipRef.current?.offsetWidth ?? 0;
    setTipW((prev) => (prev === w ? prev : w));
  }, [hover, metric, alt]);
  const tipX =
    hover == null
      ? 0
      : Math.min(Math.max(flip ? hoverX - 8 - tipW : hoverX + 8, 2), Math.max(2, width - tipW - 2));

  return (
    <div ref={wrapRef} className="relative">
      {maxTurn === 0 ? (
        <div className="flex h-[210px] items-center justify-center rounded-sm bg-chart-bg text-xs text-muted-foreground">
          No responses yet.
        </div>
      ) : (
        <svg
          width={width || 1}
          height={CHART_H}
          role="img"
          aria-label={`${metric.label} per response`}
          className="block touch-none rounded-sm bg-chart-bg"
          onPointerMove={onPointerMove}
          onPointerLeave={() => setHover(null)}
        >
          {yTicks.map((k) => {
            const y = yAt(yMax * k);
            return (
              <g key={k}>
                <line
                  x1={PAD.l}
                  x2={PAD.l + innerW}
                  y1={y}
                  y2={y}
                  className="stroke-border"
                  strokeOpacity={0.55}
                />
                <text
                  x={PAD.l - 5}
                  y={y + 3}
                  textAnchor="end"
                  fontSize={9}
                  className="fill-muted-foreground font-mono tabular-nums"
                >
                  {fmtTick(yMax * k)}
                </text>
              </g>
            );
          })}
          {xTicks.map((i) => (
            <text
              key={`x-${i}`}
              x={xAt(i)}
              y={CHART_H - 5}
              textAnchor="middle"
              fontSize={9}
              className="fill-muted-foreground font-mono tabular-nums"
            >
              {history[i].turn}
            </text>
          ))}
          <path
            d={buildPath(series)}
            fill="none"
            strokeWidth={1.75}
            strokeLinejoin="round"
            strokeLinecap="round"
            className="stroke-chart-line"
          />
          {alt ? (
            <path
              d={buildPath(alt)}
              fill="none"
              strokeWidth={1.5}
              strokeLinejoin="round"
              strokeLinecap="round"
              className="stroke-chart-alt"
            />
          ) : null}
          {buildDots(series, "fill-chart-line")}
          {alt ? buildDots(alt, "fill-chart-alt") : null}
          {hover != null ? (
            <line
              x1={hoverX}
              x2={hoverX}
              y1={PAD.t}
              y2={PAD.t + innerH}
              className="stroke-muted-foreground/60"
              strokeDasharray="3 3"
            />
          ) : null}
        </svg>
      )}

      {hover != null && maxTurn > 0 ? (
        <div
          ref={tipRef}
          className="pointer-events-none absolute top-2 z-10 min-w-36 whitespace-nowrap rounded-sm border border-border bg-popover px-2 py-1.5 font-mono text-[10px] tabular-nums shadow-[var(--shadow-border)]"
          style={{ left: tipX }}
        >
          <div className="mb-0.5 text-[9px] uppercase tracking-wide text-muted-foreground">
            Turn {history[hover].turn}
          </div>
          {(
            [
              [
                metric.tooltipLabel ?? metric.label,
                "text-chart-line",
                "bg-chart-line",
                series[hover],
              ],
              ...(alt
                ? [
                    [
                      metric.altTooltipLabel ?? ALT_LABEL,
                      "text-chart-alt",
                      "bg-chart-alt",
                      alt[hover],
                    ],
                  ]
                : []),
            ] as LegendRow[]
          ).map(([name, textCls, dotCls, v]) => (
            <div key={name} className="flex items-center gap-1.5">
              <span className={cn("size-1.5 shrink-0 rounded-full", dotCls)} />
              <span className="text-muted-foreground">{name}</span>
              <span className={cn("ml-auto pl-3 font-medium", textCls)}>
                {v != null && Number.isFinite(v) ? metric.fmt(v) : "—"}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
});

export function MetricsWidget() {
  // Shallow-primitive subscription: stable across content-only stream patches,
  // so the chart re-renders only when a response (metrics) completes.
  const samples = useChatStore(
    useShallow((s) => perfSamples(s.branches[s.currentId]?.messages ?? [])),
  );
  const history = useMemo(() => historyFromSamples(samples), [samples]);
  const [metric, setMetric] = useState<MetricDef>(METRICS[0]);

  const series = useMemo(() => metricSeries(history, metric), [history, metric]);
  const alt = useMemo(() => altSeries(history, metric), [history, metric]);
  const showAlt = alt != null && hasReasoning(history);

  const latest = lastValue(series);
  const latestAlt = alt ? lastValue(alt) : null;

  return (
    <section
      aria-label="Performance metrics"
      className="rounded-md bg-secondary/50 p-3 shadow-[var(--shadow-border)]"
    >
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        {METRICS.map((m) => (
          <button
            key={m.key}
            type="button"
            onClick={() => setMetric(m)}
            aria-pressed={metric.key === m.key}
            className={cn(
              "rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition-all duration-150",
              metric.key === m.key
                ? "border-primary bg-primary text-primary-foreground"
                : "border-border bg-secondary text-muted-foreground hover:bg-accent hover:text-foreground",
            )}
          >
            {m.label}
          </button>
        ))}
        <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground">
          {metric.unit}
        </span>
      </div>
      <PerfChart history={history} series={series} alt={showAlt ? alt : null} metric={metric} />
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] tabular-nums">
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="size-1.5 shrink-0 rounded-full bg-chart-line" />
          <span className="text-muted-foreground">{metric.label}</span>
          <span className="font-medium text-chart-line">
            {latest != null ? metric.fmt(latest) : "—"}
          </span>
        </span>
        {showAlt ? (
          <span className="flex min-w-0 items-center gap-1.5">
            <span className="size-1.5 shrink-0 rounded-full bg-chart-alt" />
            <span className="text-muted-foreground">{ALT_LABEL}</span>
            <span className="font-medium text-chart-alt">
              {latestAlt != null ? metric.fmt(latestAlt) : "—"}
            </span>
          </span>
        ) : null}
      </div>
    </section>
  );
}
