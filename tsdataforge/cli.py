from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .surface import demo as demo_bundle
from .surface import handoff as handoff_bundle
from .surface import report as report_asset


WOW_SENTENCE = (
    "Give TSDataForge one raw time-series file and it returns a report, a dataset card, a compact context, and the next actions in about one second."
)


def _emit_bundle_summary(bundle, output: Path) -> None:
    payload = {
        "output_dir": str(output),
        "wow": bundle.index.wow_sentence if bundle.index is not None else WOW_SENTENCE,
        "agent_entrypoint": "handoff_index_min.json" if bundle.index is not None else None,
        "human_open_order": bundle.index.human_open_order if bundle.index is not None else [],
        "agent_open_order": bundle.index.agent_open_order if bundle.index is not None else [],
        "recommended_next_step": bundle.index.recommended_next_step if bundle.index is not None else None,
        "why_recommended": bundle.index.why_recommended if bundle.index is not None else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _cmd_report(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = report_asset(args.input, output_path=output, docs_base_url=args.docs_base_url)
    print(report.output_path)
    return 0


def _cmd_handoff(args: argparse.Namespace) -> int:
    output = Path(args.output)
    bundle = handoff_bundle(
        args.input,
        output_dir=output,
        include_report=not args.no_report,
        include_docs_site=args.include_docs,
        include_schemas=not args.no_schemas,
        goal=args.goal,
    )
    _emit_bundle_summary(bundle, output)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    output = Path(args.output)
    bundle = demo_bundle(
        output_dir=output,
        include_docs_site=args.include_docs,
        n_series=int(args.n_series),
        length=int(args.length),
        seed=int(args.seed),
        include_schemas=not args.no_schemas,
        scenario=args.scenario,
    )
    _emit_bundle_summary(bundle, output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsdataforge",
        description="TSDataForge CLI - report and handoff workflows for time-series dataset assets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Generate a built-in demo dataset and the full report + handoff bundle.")
    p_demo.add_argument("--output", default="demo_bundle", help="Output directory for the demo bundle.")
    p_demo.add_argument("--scenario", default="ecg_public", choices=["ecg_public", "macro_public", "climate_public", "sunspots_public", "synthetic", "icu_vitals", "macro_regime", "factory_sensor"], help="Built-in demo scenario. Real public demos come first; reality-shaped synthetic demos remain available.")
    p_demo.add_argument("--include-docs", action="store_true", help="Render a local docs site into docs/ inside the bundle.")
    p_demo.add_argument("--n-series", type=int, default=24, help="Number of demo series to generate.")
    p_demo.add_argument("--length", type=int, default=192, help="Length of each demo series.")
    p_demo.add_argument("--seed", type=int, default=0, help="Random seed for the built-in demo dataset.")
    p_demo.add_argument("--no-schemas", action="store_true", help="Skip JSON Schema files. By default schemas/ is written for agent contracts.")
    p_demo.set_defaults(func=_cmd_demo)

    p_report = sub.add_parser("report", help="Generate a quick EDA report from a .npy/.npz/.csv dataset or series file.")
    p_report.add_argument("input", help="Input file (.npy, .npz, .csv, .txt, .json)")
    p_report.add_argument("--output", default="report.html", help="Output HTML file path.")
    p_report.add_argument("--docs-base-url", default=None, help="Optional base URL used for docs links inside the report.")
    p_report.set_defaults(func=_cmd_report)

    p_handoff = sub.add_parser("handoff", help="Generate the shortest report + handoff bundle for a dataset asset.")
    p_handoff.add_argument("input", help="Input file (.npy, .npz, .csv, .txt, .json)")
    p_handoff.add_argument("--output", default="tsdataforge_handoff_bundle", help="Output directory for the bundle.")
    p_handoff.add_argument("--goal", default=None, help="Optional goal that becomes part of the compact context.")
    p_handoff.add_argument("--include-docs", action="store_true", help="Render a local docs site into docs/ inside the bundle.")
    p_handoff.add_argument("--no-report", action="store_true", help="Skip report.html and only build cards/context/manifest.")
    p_handoff.add_argument("--no-schemas", action="store_true", help="Skip JSON Schema files. By default schemas/ is written for agent contracts.")
    p_handoff.set_defaults(func=_cmd_handoff)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
