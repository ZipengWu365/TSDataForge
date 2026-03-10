from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.dataset import DatasetDescription, describe_dataset
from ..analysis.describe import SeriesDescription, describe_series, suggest_spec
from ..analysis.explain import SeriesExplanation, explain_series
from ..series import GeneratedSeries
from ..agent.eda_linking import build_eda_resource_hub, save_eda_resource_hub
from . import plots


VERSION = "0.3.7"



def _json_pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return json.dumps(str(obj), indent=2, ensure_ascii=False)



def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )



def _tag_badges(tags: list[str]) -> str:
    if not tags:
        return "<span class='badge badge-muted'>no tags</span>"
    return " ".join([f"<span class='badge'>{_escape_html(t)}</span>" for t in tags])



def _metric(label: str, value: str) -> str:
    return f"<div class='metric'><div class='metric-label'>{_escape_html(label)}</div><div class='metric-value'>{_escape_html(value)}</div></div>"


def _format_value(value: Any, *, key: str | None = None) -> str:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return "n/a"
        lower_key = (key or "").lower()
        if 0.0 <= number <= 1.0 and any(token in lower_key for token in ("rate", "ratio", "spikiness")):
            return f"{number:.2%}"
        if abs(number) >= 1000:
            return f"{number:.4g}"
        if abs(number) >= 1:
            return f"{number:.3f}".rstrip("0").rstrip(".")
        if number == 0:
            return "0"
        return f"{number:.3g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(item) for item in list(value)[:4])
    return str(value)


def _summarize_evidence(evidence: Any) -> str:
    if isinstance(evidence, dict) and evidence:
        parts = []
        for key, value in list(evidence.items())[:4]:
            label = key.replace("_", " ")
            parts.append(f"{label}: {_format_value(value, key=key)}")
        return "; ".join(parts)
    return _format_value(evidence)


def _details_block(title: str, body_html: str, *, open_by_default: bool = False) -> str:
    open_attr = " open" if open_by_default else ""
    return (
        f"<details class='details-block'{open_attr}>"
        f"<summary>{_escape_html(title)}</summary>"
        f"<div class='details-body'>{body_html}</div>"
        "</details>"
    )



def _coerce_series_input(values: Any, time: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, GeneratedSeries | None]:
    if isinstance(values, GeneratedSeries):
        series = values
        arr = np.asarray(series.values, dtype=float)
        t = np.asarray(series.time if time is None else time, dtype=float)
        return arr, t, series
    arr = np.asarray(values, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError("values must be 1D or 2D, or a GeneratedSeries")
    t = np.arange(arr.shape[0], dtype=float) if time is None else np.asarray(time, dtype=float)
    return arr, t, None



def _trace_summary(series: GeneratedSeries | None) -> dict[str, Any]:
    if series is None or series.trace is None:
        return {}
    states = dict(series.trace.states)
    summary: dict[str, Any] = {
        "tags": list(series.trace.tags),
        "state_keys": sorted(states.keys()),
    }
    reward = None
    for k, v in states.items():
        if k.endswith("/reward"):
            reward = np.asarray(v, dtype=float).reshape(-1)
            summary["reward_key"] = k
            summary["episode_return"] = float(np.sum(reward))
            break
    adj = None
    for k, v in states.items():
        if k.endswith("/adjacency"):
            adj = np.asarray(v)
            summary["adjacency_key"] = k
            summary["adjacency_shape"] = list(adj.shape)
            break
    intervention_keys = [
        k for k in states
        if ("/intervention/" in k and k.endswith("/mask")) or k.endswith("/intervention_mask") or "/do_mask_var" in k
    ]
    if intervention_keys:
        summary["intervention_keys"] = intervention_keys
        counts: dict[str, int] = {}
        for k in intervention_keys:
            arr = np.asarray(states[k])
            if arr.ndim == 1:
                counts[k] = int(np.sum(arr > 0))
        summary["intervention_counts"] = counts
    cf_keys = sorted(k for k in states if "/counterfactual/" in k and k.endswith("/output"))
    if cf_keys:
        summary["counterfactual_keys"] = cf_keys
        returns = {k: float(v) for k, v in states.items() if "/counterfactual/" in k and k.endswith("/return")}
        if returns:
            summary["counterfactual_returns"] = returns
    if any(k.endswith("/potential_outcome_do0") for k in states):
        summary["has_potential_outcomes"] = True
    return summary



def _render_trace_card(trace_summary: dict[str, Any]) -> str:
    if not trace_summary:
        return ""
    bullets: list[str] = []
    if "episode_return" in trace_summary:
        bullets.append(f"Observed episode return: {trace_summary['episode_return']:.4g}.")
    if "adjacency_shape" in trace_summary:
        bullets.append(f"Causal graph available with shape {tuple(trace_summary['adjacency_shape'])}.")
    if trace_summary.get("counterfactual_keys"):
        bullets.append(f"Stored {len(trace_summary['counterfactual_keys'])} counterfactual rollout(s).")
    if trace_summary.get("intervention_counts"):
        counts = ", ".join([f"{k.split('/')[-1]}={v}" for k, v in list(trace_summary["intervention_counts"].items())[:6]])
        bullets.append(f"Intervention activity detected: {counts}.")
    if trace_summary.get("has_potential_outcomes"):
        bullets.append("Potential outcomes and ITE traces are available.")
    if not bullets:
        bullets.append("Trace metadata is available but no control/causal extras were detected.")
    details = _details_block(
        "Show trace metadata (JSON)",
        f"<pre><code>{_escape_html(_json_pretty(trace_summary))}</code></pre>",
    )
    return (
        "<div class='card'>"
        "<h2>Control / causal trace summary</h2>"
        "<ul>"
        + "".join([f"<li>{_escape_html(b)}</li>" for b in bullets])
        + "</ul>"
        + details
        + "</div>"
    )



def _resource_links_html(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    rows = []
    for item in items:
        title_text = getattr(item, "title", None) or ""
        path_text = getattr(item, "path", None) or ""
        reason_text = getattr(item, "reason", None) or ""
        rows.append(f"<li><a href='{_escape_html(path_text)}'>{_escape_html(title_text)}</a><div class='muted'>{_escape_html(reason_text)}</div></li>")
    return "<div class='card'><h2>" + _escape_html(title) + "</h2><ul class='link-list'>" + ''.join(rows) + "</ul></div>"


def _render_resource_hub_card(resource_hub: Any | None) -> str:
    if resource_hub is None:
        return ""
    task_items = ""
    if getattr(resource_hub, "recommended_tasks", None):
        task_rows = []
        for item in resource_hub.recommended_tasks:
            task_rows.append("<tr><td><span class='badge'>" + _escape_html(item.task) + "</span></td><td>" + _escape_html(item.reason) + "</td></tr>")
        task_items = (
            "<div class='card'><h2>Recommended tasks from this report</h2>"
            "<table class='table'><thead><tr><th>Task</th><th>Why now</th></tr></thead><tbody>"
            + ''.join(task_rows)
            + "</tbody></table></div>"
        )
    routes_html = ""
    if getattr(resource_hub, "routes", None):
        route_badges = ''.join([f"<span class='badge'>{_escape_html(route.title)}</span>" for route in resource_hub.routes])
        routes_html = "<div class='card'><h2>Matched report routes</h2><div class='tags'>" + route_badges + "</div><p class='muted'>These routes are the shared bridge used by the report and the docs site.</p></div>"
    next_steps_html = ""
    if getattr(resource_hub, "next_steps", None):
        next_steps_html = "<div class='card'><h2>Next steps</h2><ul>" + ''.join([f"<li>{_escape_html(item)}</li>" for item in resource_hub.next_steps]) + "</ul></div>"
    return (
        "<div class='card'><h2>Next actions and useful links</h2><p class='muted'>This report can route you from the main findings to docs, runnable examples, API entry points, and FAQ answers.</p></div>"
        + task_items
        + routes_html
        + next_steps_html
        + "<div class='link-grid'>"
        + _resource_links_html("Docs pages", getattr(resource_hub, "page_links", []))
        + _resource_links_html("Examples", getattr(resource_hub, "example_links", []))
        + _resource_links_html("API entry points", getattr(resource_hub, "api_links", []))
        + _resource_links_html("FAQ matches", getattr(resource_hub, "faq_links", []))
        + "</div>"
    )


def _render_series_report(
    desc: SeriesDescription,
    expl: SeriesExplanation,
    *,
    values: np.ndarray,
    time: np.ndarray,
    channel_names: list[str] | None,
    title: str,
    include_spec: bool,
    trace_summary: dict[str, Any] | None,
    resource_hub: Any | None,
    series: GeneratedSeries | None,
) -> str:
    plot_blocks: list[str] = []
    plot_errors: list[str] = []
    try:
        plot_blocks.append(
            f"<div class='plot'><img src='data:image/png;base64,{plots.plot_series_overview(values, time, channel_names=channel_names)}'/></div>"
        )
        plot_blocks.append(
            f"<div class='plot'><img src='data:image/png;base64,{plots.plot_missingness(values, time)}'/></div>"
        )
        plot_blocks.append(
            f"<div class='plot'><img src='data:image/png;base64,{plots.plot_histogram(values)}'/></div>"
        )
        plot_blocks.append(
            f"<div class='plot'><img src='data:image/png;base64,{plots.plot_power_spectrum(values, time)}'/></div>"
        )
        plot_blocks.append(
            f"<div class='plot'><img src='data:image/png;base64,{plots.plot_acf(values)}'/></div>"
        )
        if desc.n_channels > 1:
            plot_blocks.append(
                f"<div class='plot'><img src='data:image/png;base64,{plots.plot_cross_correlation(values)}'/></div>"
            )
        if desc.dt_cv > 0.05:
            plot_blocks.append(
                f"<div class='plot'><img src='data:image/png;base64,{plots.plot_sampling_intervals(time)}'/></div>"
            )
        period = None
        if desc.scores and "dominant_period" in desc.scores and np.isfinite(desc.scores["dominant_period"]):
            period = float(desc.scores["dominant_period"])
        plot_blocks.append(
            f"<div class='plot'><img src='data:image/png;base64,{plots.plot_decomposition(values, time, period=period)}'/></div>"
        )
        if series is not None and series.trace is not None:
            states = dict(series.trace.states)
            for k, v in states.items():
                if k.endswith("/adjacency"):
                    plot_blocks.append(
                        f"<div class='plot'><img src='data:image/png;base64,{plots.plot_adjacency_matrix(np.asarray(v))}'/></div>"
                    )
                    break
            cf_keys = sorted(k for k in states if "/counterfactual/" in k and k.endswith("/output"))
            if cf_keys:
                cf = np.asarray(states[cf_keys[0]], dtype=float)
                label = cf_keys[0].split("/counterfactual/")[-1].split("/")[0]
                plot_blocks.append(
                    f"<div class='plot'><img src='data:image/png;base64,{plots.plot_factual_vs_counterfactual(values, cf, time, label=label)}'/></div>"
                )
            iv_keys = [
                k for k in states
                if ("/intervention/" in k and k.endswith("/mask")) or k.endswith("/intervention_mask") or "/do_mask_var" in k
            ]
            if iv_keys:
                mask = np.zeros(len(time), dtype=int)
                for key in iv_keys:
                    arr = np.asarray(states[key])
                    if arr.ndim == 1 and len(arr) == len(time):
                        mask = np.maximum(mask, (arr > 0).astype(int))
                plot_blocks.append(
                    f"<div class='plot'><img src='data:image/png;base64,{plots.plot_binary_mask(time, mask, title='Intervention activity')}'/></div>"
                )
    except Exception as e:  # pragma: no cover
        plot_errors.append(str(e))

    spec_block = ""
    if include_spec:
        try:
            spec = suggest_spec(desc)
            spec_block = _details_block(
                "Suggested synthetic seed",
                (
                    "<p class='muted'>This is a coarse, explainable seed derived from the report. "
                    "Use it when you need a matching synthetic control or a benchmark starter.</p>"
                    f"<pre><code>{_escape_html(_json_pretty(spec.to_dict()))}</code></pre>"
                ),
            )
        except Exception as e:  # pragma: no cover
            spec_block = _details_block(
                "Suggested synthetic seed",
                f"<p class='warn'>Failed to suggest spec: {_escape_html(str(e))}</p>",
            )

    metric_items = [
        _metric("Length", str(desc.length)),
        _metric("Channels", str(desc.n_channels)),
        _metric("Missing rate", f"{desc.missing_rate:.2%}"),
        _metric("dt mean", f"{desc.dt_mean:.4g}"),
        _metric("dt CV", f"{desc.dt_cv:.3f}"),
    ]
    if trace_summary:
        if "episode_return" in trace_summary:
            metric_items.append(_metric("Episode return", f"{trace_summary['episode_return']:.4g}"))
        if trace_summary.get("counterfactual_keys"):
            metric_items.append(_metric("Counterfactuals", str(len(trace_summary["counterfactual_keys"]))))
        if trace_summary.get("intervention_keys"):
            metric_items.append(_metric("Interventions", str(len(trace_summary["intervention_keys"]))))
    metrics_html = "".join(metric_items)

    insight_cards = []
    for tag, msg in expl.tag_explanations.items():
        ev = expl.evidence.get(tag, {})
        summary = _summarize_evidence(ev) if ev else "No structured metrics captured."
        metrics_block = f"<p class='small'><strong>Metrics:</strong> {_escape_html(summary)}</p>"
        details_block = ""
        if ev:
            details_block = _details_block(
                "Show raw metrics",
                f"<pre><code>{_escape_html(_json_pretty(ev))}</code></pre>",
            )
        insight_cards.append(
            "<div class='insight-card'>"
            f"<div class='tags'><span class='badge'>{_escape_html(tag)}</span></div>"
            f"<p>{_escape_html(msg)}</p>"
            + metrics_block
            + details_block
            + "</div>"
        )
    insights_html = (
        "<div class='insight-grid'>" + "".join(insight_cards) + "</div>"
        if insight_cards
        else "<p class='muted'>No structure-specific notes are available for this series.</p>"
    )

    plot_error_html = ""
    if plot_errors:
        plot_error_html = "<div class='warn'>Plotting failed: <pre><code>{}</code></pre></div>".format(
            _escape_html("\n".join(plot_errors))
        )

    trace_card = _render_trace_card(trace_summary or {})
    appendix_blocks = [
        _details_block(
            "Raw series description (JSON)",
            f"<pre><code>{_escape_html(_json_pretty(desc.to_dict()))}</code></pre>",
        ),
        _details_block(
            "Explanation payload (JSON)",
            f"<pre><code>{_escape_html(_json_pretty(expl.to_dict()))}</code></pre>",
        ),
    ]
    if spec_block:
        appendix_blocks.append(spec_block)
    html = f"""
    <div class='container'>
      <div class='hero'>
        <div class='eyebrow'>Human-readable report</div>
        <h1>{_escape_html(title)}</h1>
        <p class='subtitle'>{_escape_html(expl.headline)}</p>
        <div class='tags'>{_tag_badges(desc.inferred_tags)}</div>
      </div>

      <div class='card'>
        <h2>Quick facts</h2>
        <div class='metrics'>{metrics_html}</div>
      </div>

      <div class='card'>
        <h2>What stands out</h2>
        <ul>
          {''.join([f"<li>{_escape_html(b)}</li>" for b in expl.bullets])}
        </ul>
      </div>

      <div class='card'>
        <h2>What the structure suggests</h2>
        <p class='muted'>Use these notes to decide whether the series looks stable, seasonal, bursty, irregular, or worth routing into a specific task.</p>
        {insights_html}
      </div>

      {trace_card}

      {_render_resource_hub_card(resource_hub)}

      <div class='card'>
        <h2>Visual diagnostics</h2>
        {plot_error_html}
        <div class='plot-grid'>
          {''.join(plot_blocks)}
        </div>
      </div>

      <div class='card'>
        <h2>Technical appendix</h2>
        <p class='muted'>These structured details are useful for debugging, reproducibility, or agent handoff. They are kept out of the main reading flow on purpose.</p>
        {''.join(appendix_blocks)}
      </div>

      <div class='footer'>Generated by tsdataforge.report (v{VERSION}). Pure numpy/scipy analysis; plots via matplotlib when available.</div>
    </div>
    """
    return html



def _render_dataset_report(
    ddesc: DatasetDescription,
    *,
    title: str,
    resource_hub: Any | None,
) -> str:
    plot_blocks: list[str] = []
    plot_errors: list[str] = []
    try:
        plot_blocks.append(
            f"<div class='plot'><img src='data:image/png;base64,{plots.plot_tag_frequency(ddesc.tag_counts)}'/></div>"
        )
        if ddesc.per_series is not None:
            lengths = [float(s.length) for s in ddesc.per_series]
            missing = [float(s.missing_rate) for s in ddesc.per_series]
            dtcv = [float(s.dt_cv) for s in ddesc.per_series]
            chans = [float(s.n_channels) for s in ddesc.per_series]
            plot_blocks.append(
                f"<div class='plot'><img src='data:image/png;base64,{plots.plot_feature_hist(lengths, 'Series length distribution')}'/></div>"
            )
            plot_blocks.append(
                f"<div class='plot'><img src='data:image/png;base64,{plots.plot_feature_hist(chans, 'Channel count distribution')}'/></div>"
            )
            plot_blocks.append(
                f"<div class='plot'><img src='data:image/png;base64,{plots.plot_feature_hist(missing, 'Missing rate distribution')}'/></div>"
            )
            plot_blocks.append(
                f"<div class='plot'><img src='data:image/png;base64,{plots.plot_feature_hist(dtcv, 'Sampling irregularity (dt CV) distribution')}'/></div>"
            )
    except Exception as e:  # pragma: no cover
        plot_errors.append(str(e))

    plot_error_html = ""
    if plot_errors:
        plot_error_html = "<div class='warn'>Plotting failed: <pre><code>{}</code></pre></div>".format(
            _escape_html("\n".join(plot_errors))
        )

    n = int(ddesc.n_series)
    top_tags = list(ddesc.tag_counts.items())[:8]
    top_sigs = list(ddesc.signature_counts.items())[:8]
    bullets = []
    if top_tags:
        bullets.append(
            "Most frequent tags: "
            + ", ".join([f"{k} ({v}/{n})" for k, v in top_tags])
            + "."
        )
    if top_sigs:
        bullets.append(
            "Most common structure signatures: "
            + ", ".join([f"{k} ({v})" for k, v in top_sigs])
            + "."
        )
    bullets.append(
        "Signatures are a compact subset of tags (trend/seasonal/ar1/random_walk/bursty + observation/multivariate/coupled) for quick dataset-level grouping."
    )

    metrics_html = "".join(
        [
            _metric("Series count", str(ddesc.n_series)),
            _metric("Length (median)", f"{ddesc.length_stats.get('median', float('nan')):.0f}"),
            _metric("Channels (median)", f"{ddesc.channel_stats.get('median', float('nan')):.0f}"),
            _metric("Missing rate (median)", f"{ddesc.missing_rate_stats.get('median', float('nan')):.2%}"),
            _metric("dt CV (median)", f"{ddesc.dt_cv_stats.get('median', float('nan')):.3f}"),
        ]
    )
    appendix_html = _details_block(
        "Raw dataset description (JSON)",
        f"<pre><code>{_escape_html(_json_pretty(ddesc.to_dict()))}</code></pre>",
    )

    html = f"""
    <div class='container'>
      <div class='hero'>
        <div class='eyebrow'>Human-readable report</div>
        <h1>{_escape_html(title)}</h1>
        <p class='subtitle'>Dataset-level EDA focused on structure, sampling quality, coverage, and sensible next steps.</p>
      </div>

      <div class='card'>
        <h2>Quick facts</h2>
        <div class='metrics'>{metrics_html}</div>
      </div>

      <div class='card'>
        <h2>Highlights</h2>
        <ul>
          {''.join([f"<li>{_escape_html(b)}</li>" for b in bullets])}
        </ul>
      </div>

      {_render_resource_hub_card(resource_hub)}

      <div class='card'>
        <h2>Visual diagnostics</h2>
        {plot_error_html}
        <div class='plot-grid'>
          {''.join(plot_blocks)}
        </div>
      </div>

      <div class='card'>
        <h2>Technical appendix</h2>
        <p class='muted'>Structured payloads are kept here so the main report stays readable.</p>
        {appendix_html}
      </div>

      <div class='footer'>Generated by tsdataforge.report (v{VERSION}).</div>
    </div>
    """
    return html


_CSS = """
:root{--bg:#ffffff;--bg-muted:#f5f7fa;--panel:#ffffff;--panel-muted:#f8fafc;--text:#16202a;--muted:#5c6775;--accent:#f89939;--accent-soft:#fff3e5;--link:#1f5fbf;--border:#d9dee5;--border-strong:#c7cfd9;--shadow:0 6px 18px rgba(15,23,42,.06);--code-bg:#f7f8fa;}
*{box-sizing:border-box;}
body{margin:0;font-family:"Segoe UI",Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);}
.container{max-width:1120px;margin:0 auto;padding:28px 18px 48px;}
.hero{border:1px solid var(--border);border-radius:10px;padding:24px 24px 20px;background:var(--panel-muted);box-shadow:var(--shadow);margin-bottom:14px;}
.eyebrow{color:var(--muted);text-transform:uppercase;font-weight:700;font-size:12px;letter-spacing:.08em;margin-bottom:8px;}
h1{font-size:34px;line-height:1.14;margin:0 0 8px;}
h2{font-size:26px;line-height:1.2;margin:0 0 10px;}
h3{font-size:19px;line-height:1.3;margin:0 0 8px;}
.subtitle{margin:0;color:var(--muted);font-size:17px;max-width:840px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:18px;margin:14px 0;box-shadow:var(--shadow);}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:12px;}
.metric{background:var(--panel-muted);border:1px solid var(--border);border-radius:10px;padding:12px 13px;}
.metric-label{font-size:12px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em;}
.metric-value{font-size:20px;font-weight:700;color:var(--text);}
.tags{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;}
.badge{display:inline-flex;align-items:center;padding:5px 10px;border-radius:999px;background:var(--accent-soft);border:1px solid #f0c18c;color:#8a4d00;font-size:12px;}
.badge-muted{background:var(--panel-muted);border-color:var(--border);color:var(--muted);}
.insight-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:14px;}
.insight-card{background:var(--panel-muted);border:1px solid var(--border);border-radius:10px;padding:14px;}
.plot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;}
.plot img{width:100%;border-radius:10px;border:1px solid var(--border);background:var(--panel);}
.table{width:100%;border-collapse:collapse;margin-top:10px;border:1px solid var(--border);}
.table th,.table td{border-bottom:1px solid var(--border);padding:11px 10px;vertical-align:top;text-align:left;}
.table th{background:var(--panel-muted);font-weight:700;color:var(--text);}
.link-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;}
.link-list{padding-left:18px;margin:10px 0 0;}
.link-list li{margin-bottom:10px;}
a{color:var(--link);text-decoration:none;}
a:hover{text-decoration:underline;}
pre{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;padding:14px;overflow:auto;color:#0f172a;}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;background:rgba(15,23,42,.04);border-radius:4px;padding:0 4px;color:#0f172a;}
pre code{background:transparent;padding:0;}
.details-block{margin-top:12px;border:1px solid var(--border);border-radius:8px;background:var(--panel);}
.details-block summary{cursor:pointer;padding:12px 14px;font-weight:700;color:var(--text);}
.details-body{padding:0 14px 14px;}
.warn{background:#fff7e8;border:1px solid #f0c18c;border-radius:10px;padding:10px 12px;color:#8a4d00;}
.muted,.small{color:var(--muted);}
.small{font-size:13px;}
strong{color:var(--text);}
.footer{margin-top:18px;color:var(--muted);font-size:12px;text-align:center;}
ul{padding-left:22px;}
li{margin-bottom:8px;}
@media (max-width:720px){.container{padding:18px 14px 40px;}.hero{padding:20px;}.hero h1,h1{font-size:29px;}}
"""


@dataclass
class EDAReport:
    """An HTML EDA report."""

    html: str
    output_path: str | None
    kind: str
    summary: dict[str, Any]
    resource_hub: Any | None = None

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.html, encoding="utf-8")
        self.output_path = str(p)
        if self.resource_hub is not None:
            try:
                save_eda_resource_hub(self.resource_hub, p)
            except Exception:
                pass
        return str(p)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "kind": self.kind,
            "output_path": self.output_path,
            "summary": self.summary,
        }
        if self.resource_hub is not None:
            try:
                out["resource_hub"] = self.resource_hub.to_dict()
            except Exception:
                out["resource_hub"] = str(self.resource_hub)
        return out



def generate_eda_report(
    values: Any,
    time: np.ndarray | None = None,
    *,
    channel_names: list[str] | None = None,
    title: str = "TSDataForge EDA Report",
    output_path: str | Path | None = "tsdataforge_report.html",
    include_suggested_spec: bool = True,
    docs_base_url: str | None = None,
    include_linked_resources: bool = True,
) -> EDAReport:
    """Generate a single-series EDA report (HTML).

    `values` may be a raw array `(T,)` / `(T,C)` or a `GeneratedSeries`.
    When a `GeneratedSeries` is provided, the report incorporates trace-aware
    control / causal summaries and extra visuals.
    """

    arr, t, series = _coerce_series_input(values, time)
    desc = describe_series(arr, t)
    expl = explain_series(desc)
    trace_summary = _trace_summary(series)
    resource_hub = build_eda_resource_hub(desc, trace_summary=trace_summary, docs_base_url=docs_base_url) if include_linked_resources else None
    body = _render_series_report(
        desc,
        expl,
        values=arr,
        time=t,
        channel_names=channel_names,
        title=title,
        include_spec=include_suggested_spec,
        trace_summary=trace_summary,
        resource_hub=resource_hub,
        series=series,
    )
    full = f"<!doctype html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/><title>{_escape_html(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>"
    report = EDAReport(
        html=full,
        output_path=None,
        kind="series",
        summary={
            "description": desc.to_dict(),
            "explanation": expl.to_dict(),
            "trace_summary": trace_summary,
            "resource_hub": resource_hub.to_dict() if resource_hub is not None else None,
        },
        resource_hub=resource_hub,
    )
    if output_path is not None:
        report.save(output_path)
    return report



def generate_dataset_eda_report(
    values: Any,
    time: Any | None = None,
    *,
    title: str = "TSDataForge Dataset EDA Report",
    output_path: str | Path | None = "tsdataforge_dataset_report.html",
    max_series: int | None = 200,
    seed: int = 0,
    docs_base_url: str | None = None,
    include_linked_resources: bool = True,
) -> EDAReport:
    """Generate a dataset-level EDA report (HTML)."""

    ddesc = describe_dataset(values, time, max_series=max_series, seed=seed)
    resource_hub = build_eda_resource_hub(ddesc, docs_base_url=docs_base_url) if include_linked_resources else None
    body = _render_dataset_report(ddesc, title=title, resource_hub=resource_hub)
    full = f"<!doctype html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/><title>{_escape_html(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>"
    report = EDAReport(
        html=full,
        output_path=None,
        kind="dataset",
        summary={"dataset_description": ddesc.to_dict(), "resource_hub": resource_hub.to_dict() if resource_hub is not None else None},
        resource_hub=resource_hub,
    )
    if output_path is not None:
        report.save(output_path)
    return report



def generate_linked_eda_bundle(
    values: Any,
    time: np.ndarray | None = None,
    *,
    output_dir: str | Path = "tsdataforge_eda_bundle",
    title: str = "TSDataForge Linked EDA Report",
    docs_title: str = "TSDataForge Docs",
    channel_names: list[str] | None = None,
    include_suggested_spec: bool = True,
) -> dict[str, str]:
    """Generate a shareable bundle with a linked EDA report and docs site."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs_dir = out / "docs"
    from ..agent.site import generate_docs_site

    site = generate_docs_site(docs_dir, title=docs_title)
    generate_eda_report(
        values,
        time,
        channel_names=channel_names,
        title=title,
        output_path=out / "report.html",
        include_suggested_spec=include_suggested_spec,
        docs_base_url="docs/",
        include_linked_resources=True,
    )
    manifest = {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "docs_site": site.to_dict(),
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# TSDataForge Linked EDA Bundle\n\n- 打开 `report.html` 查看带推荐链接的 EDA 报告。\n- 从报告里可直接跳到 `docs/` 下的 landing / quickstart / cookbook / API / FAQ。\n- 该目录可直接整体分享。\n",
        encoding="utf-8",
    )
    return {"report": str(out / "report.html"), "docs_index": str(docs_dir / "index.html"), "bundle_manifest": str(out / "bundle_manifest.json")}


def generate_linked_dataset_eda_bundle(
    values: Any,
    time: Any | None = None,
    *,
    output_dir: str | Path = "tsdataforge_dataset_eda_bundle",
    title: str = "TSDataForge Linked Dataset EDA Report",
    docs_title: str = "TSDataForge Docs",
    max_series: int | None = 200,
    seed: int = 0,
) -> dict[str, str]:
    """Generate a shareable dataset-level bundle with report + docs site."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs_dir = out / "docs"
    from ..agent.site import generate_docs_site

    site = generate_docs_site(docs_dir, title=docs_title)
    generate_dataset_eda_report(
        values,
        time,
        title=title,
        output_path=out / "report.html",
        max_series=max_series,
        seed=seed,
        docs_base_url="docs/",
        include_linked_resources=True,
    )
    manifest = {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "docs_site": site.to_dict(),
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# TSDataForge Linked Dataset EDA Bundle\n\n"
        "- Open `report.html` for the dataset-level report.\n"
        "- Use links inside the report to jump to `docs/` pages, examples, API, and FAQ.\n"
        "- Share this whole folder as a portable bundle.\n",
        encoding="utf-8",
    )
    return {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "bundle_manifest": str(out / "bundle_manifest.json"),
    }


def generate_linked_eda_bundle(
    values: Any,
    time: np.ndarray | None = None,
    *,
    output_dir: str | Path = "tsdataforge_eda_bundle",
    title: str = "TSDataForge Linked EDA Report",
    docs_title: str = "TSDataForge Docs",
    channel_names: list[str] | None = None,
    include_suggested_spec: bool = True,
) -> dict[str, str]:
    """Generate a shareable bundle with a linked EDA report and docs site."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs_dir = out / "docs"
    from ..agent.site import generate_docs_site

    site = generate_docs_site(docs_dir, title=docs_title)
    generate_eda_report(
        values,
        time,
        channel_names=channel_names,
        title=title,
        output_path=out / "report.html",
        include_suggested_spec=include_suggested_spec,
        docs_base_url="docs/",
        include_linked_resources=True,
    )
    manifest = {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "docs_site": site.to_dict(),
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# TSDataForge Linked EDA Bundle\n\n"
        "- Open `report.html` for the human-readable report.\n"
        "- Use links inside the report to jump to `docs/` pages, examples, API, and FAQ.\n"
        "- Share this whole folder as a portable bundle.\n",
        encoding="utf-8",
    )
    return {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "bundle_manifest": str(out / "bundle_manifest.json"),
    }


def generate_linked_dataset_eda_bundle(
    values: Any,
    time: Any | None = None,
    *,
    output_dir: str | Path = "tsdataforge_dataset_eda_bundle",
    title: str = "TSDataForge Linked Dataset EDA Report",
    docs_title: str = "TSDataForge Docs",
    max_series: int | None = 200,
    seed: int = 0,
) -> dict[str, str]:
    """Generate a shareable dataset-level bundle with report + docs site."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs_dir = out / "docs"
    from ..agent.site import generate_docs_site

    site = generate_docs_site(docs_dir, title=docs_title)
    generate_dataset_eda_report(
        values,
        time,
        title=title,
        output_path=out / "report.html",
        max_series=max_series,
        seed=seed,
        docs_base_url="docs/",
        include_linked_resources=True,
    )
    manifest = {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "docs_site": site.to_dict(),
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# TSDataForge Linked Dataset EDA Bundle\n\n"
        "- Open `report.html` for the dataset-level report.\n"
        "- Use links inside the report to jump to `docs/` pages, examples, API, and FAQ.\n"
        "- Share this whole folder as a portable bundle.\n",
        encoding="utf-8",
    )
    return {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "bundle_manifest": str(out / "bundle_manifest.json"),
    }



def generate_linked_dataset_eda_bundle(
    values: Any,
    time: Any | None = None,
    *,
    output_dir: str | Path = "tsdataforge_dataset_eda_bundle",
    title: str = "TSDataForge Linked Dataset EDA Report",
    docs_title: str = "TSDataForge Docs",
    max_series: int | None = 200,
    seed: int = 0,
) -> dict[str, str]:
    """Generate a shareable dataset-level bundle with report + docs site."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs_dir = out / "docs"
    from ..agent.site import generate_docs_site

    site = generate_docs_site(docs_dir, title=docs_title)
    generate_dataset_eda_report(
        values,
        time,
        title=title,
        output_path=out / "report.html",
        max_series=max_series,
        seed=seed,
        docs_base_url="docs/",
        include_linked_resources=True,
    )
    manifest = {
        "report": str(out / "report.html"),
        "docs_index": str(docs_dir / "index.html"),
        "docs_site": site.to_dict(),
    }
    (out / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# TSDataForge Linked Dataset EDA Bundle\n\n- 打开 `report.html` 查看数据集级 EDA 报告。\n- 报告中的建议会直接跳到 `docs/` 里的 taskification / cookbook / API / FAQ。\n",
        encoding="utf-8",
    )
    return {"report": str(out / "report.html"), "docs_index": str(docs_dir / "index.html"), "bundle_manifest": str(out / "bundle_manifest.json")}
