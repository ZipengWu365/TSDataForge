from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from tsdataforge import demo
from tsdataforge.cli import build_parser
from tsdataforge.gui import build_gui_html, bundle_summary_for_gui, create_gui_app


def test_cli_accepts_ui_command():
    args = build_parser().parse_args(["ui", "--host", "127.0.0.1", "--port", "9001", "--output-root", ".bundle/gui"])
    assert args.command == "ui"
    assert args.host == "127.0.0.1"
    assert args.port == 9001
    assert args.output_root == ".bundle/gui"


def test_build_gui_html_contains_upload_and_demo_paths():
    html = build_gui_html()
    assert "Drag a .npy, .npz, .csv, .txt, or .json file here" in html
    assert 'fetch("/api/handoff"' in html
    assert 'fetch("/api/demo"' in html
    assert "TSDataForge Local GUI" in html


def test_gui_server_serves_index_and_health(tmp_path: Path):
    app = create_gui_app(port=0, output_root=tmp_path / "gui_runs")
    worker = threading.Thread(target=app.serve_forever, daemon=True)
    worker.start()
    try:
        with urllib.request.urlopen(f"{app.url}/healthz") as response:
            payload = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(app.url) as response:
            html = response.read().decode("utf-8")
        assert payload["ok"] is True
        assert "TSDataForge Local GUI" in html
    finally:
        app.shutdown()
        worker.join(timeout=5)


def test_bundle_summary_for_gui_uses_agent_first_entry(tmp_path: Path):
    pytest.importorskip("matplotlib")
    bundle = demo(output_dir=tmp_path / "demo_bundle", scenario="ecg_public")
    summary = bundle_summary_for_gui(bundle, run_id="run-123", origin="http://127.0.0.1:8765", input_name="ecg_public")
    assert summary["agent_entrypoint"] == "handoff_index_min.json"
    assert summary["links"]["report"] == "http://127.0.0.1:8765/runs/run-123/report.html"
    assert summary["links"]["index_min"] == "http://127.0.0.1:8765/runs/run-123/handoff_index_min.json"
    assert summary["links"]["card"] == "http://127.0.0.1:8765/runs/run-123/dataset_card.md"
    assert "report.html" in summary["human_open_order"]


def test_gui_upload_endpoint_builds_bundle(tmp_path: Path):
    pytest.importorskip("matplotlib")
    app = create_gui_app(port=0, output_root=tmp_path / "gui_runs")
    worker = threading.Thread(target=app.serve_forever, daemon=True)
    worker.start()
    sample = tmp_path / "sample.npy"
    np.save(sample, np.linspace(0.0, 1.0, 64))

    boundary = "tsdataforge-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="goal"\r\n\r\n'
        f"decide the next downstream task\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{sample.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") + sample.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")

    request = urllib.request.Request(
        f"{app.url}/api/handoff",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["links"]["report"].endswith("/report.html")
        assert payload["links"]["index_min"].endswith("/handoff_index_min.json")
        assert payload["recommended_next_step"] is not None
    finally:
        app.shutdown()
        worker.join(timeout=5)
