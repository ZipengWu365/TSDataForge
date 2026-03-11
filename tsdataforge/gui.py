from __future__ import annotations

import json
import mimetypes
import os
import secrets
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


DEFAULT_GUI_HOST = "127.0.0.1"
DEFAULT_GUI_PORT = 8765
DEFAULT_GUI_OUTPUT_ROOT = Path(".bundle/gui_runs")
DEMO_SCENARIOS = (
    "ecg_public",
    "macro_public",
    "climate_public",
    "sunspots_public",
    "synthetic",
    "icu_vitals",
    "macro_regime",
    "factory_sensor",
)


def _slugify_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
    cleaned = cleaned.strip("._")
    return cleaned or "upload.bin"


def _run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def _safe_route(*parts: str) -> str:
    return "/".join(quote(part) for part in parts if part)


def _safe_child(root: Path, relative: Path) -> Path:
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath([str(root_resolved), str(candidate)]) != str(root_resolved):
        raise PermissionError("Requested path escapes the GUI output root.")
    return candidate


def _file_href(origin: str, run_id: str, relative_name: str) -> str:
    parts = Path(relative_name.rstrip("/")).parts
    return f"{origin}/runs/{_safe_route(run_id, *parts)}"


def _parse_multipart_form(content_type: str, body: bytes) -> dict[str, dict[str, Any]]:
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    fields: dict[str, dict[str, Any]] = {}
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        fields[str(name)] = {
            "filename": part.get_filename(),
            "value": part.get_content() if part.get_filename() is None else None,
            "data": payload,
        }
    return fields


def bundle_summary_for_gui(bundle: Any, *, run_id: str, origin: str, input_name: str | None = None) -> dict[str, Any]:
    index = bundle.index
    artifact_names = {item.name for item in bundle.artifacts}

    def maybe_link(name: str) -> str | None:
        if name not in artifact_names:
            return None
        return _file_href(origin, run_id, name)

    report_name = "report.html" if "report.html" in artifact_names else None
    card_name = "dataset_card.md" if "dataset_card.md" in artifact_names else "task_card.md" if "task_card.md" in artifact_names else None
    context_name = "dataset_context.json" if "dataset_context.json" in artifact_names else "task_context.json" if "task_context.json" in artifact_names else None

    links = {
        "report": maybe_link(report_name) if report_name else None,
        "card": maybe_link(card_name) if card_name else None,
        "context": maybe_link(context_name) if context_name else None,
        "index_min": maybe_link("handoff_index_min.json"),
        "index": maybe_link("handoff_index.json"),
        "action_plan": maybe_link("action_plan.json"),
        "bundle": maybe_link("handoff_bundle.json"),
        "resource_hub": maybe_link("report_resource_hub.md"),
        "readme": maybe_link("README.md"),
    }

    artifacts: list[dict[str, Any]] = []
    for item in bundle.artifacts:
        href = None if item.name.endswith("/") else _file_href(origin, run_id, item.name)
        artifacts.append(
            {
                "name": item.name,
                "kind": item.kind,
                "description": item.description,
                "href": href,
            }
        )

    return {
        "run_id": run_id,
        "dataset_id": bundle.dataset_id,
        "title": bundle.title,
        "summary": bundle.summary,
        "wow_sentence": index.wow_sentence if index is not None else None,
        "recommended_next_step": index.recommended_next_step if index is not None else None,
        "why_recommended": index.why_recommended if index is not None else None,
        "human_open_order": index.human_open_order if index is not None else [],
        "agent_open_order": index.agent_open_order if index is not None else [],
        "agent_entrypoint": "handoff_index_min.json" if index is not None else None,
        "recommended_prompt": index.recommended_prompt if index is not None else None,
        "links": links,
        "artifacts": artifacts,
        "output_dir": bundle.output_dir,
        "input_name": input_name,
    }


def build_gui_html() -> str:
    scenario_options = "\n".join(
        f"<option value='{scenario}'>{scenario}</option>" for scenario in DEMO_SCENARIOS
    )
    scenarios_json = json.dumps(list(DEMO_SCENARIOS))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TSDataForge Local GUI</title>
  <style>
    :root {{
      --bg: #f6f1e7;
      --paper: rgba(255, 252, 245, 0.92);
      --panel: rgba(255, 255, 255, 0.88);
      --ink: #132033;
      --muted: #5d6876;
      --line: rgba(19, 32, 51, 0.12);
      --accent: #0f766e;
      --accent-2: #c2410c;
      --accent-3: #1d4ed8;
      --shadow: 0 18px 48px rgba(19, 32, 51, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(194, 65, 12, 0.10), transparent 22%),
        linear-gradient(180deg, #f8f4eb 0%, #f3ecde 100%);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 18px 56px;
    }}
    .hero {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 28px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
      font-weight: 700;
      color: var(--accent);
    }}
    h1 {{
      font-family: "IBM Plex Serif", Georgia, serif;
      font-size: clamp(36px, 6vw, 58px);
      line-height: 0.98;
      margin: 12px 0;
      max-width: 780px;
    }}
    .hero p {{
      max-width: 860px;
      font-size: 18px;
      color: var(--muted);
      margin: 0;
    }}
    .hero-grid {{
      display: grid;
      grid-template-columns: 1.06fr 0.94fr;
      gap: 18px;
      margin-top: 24px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .panel h2, .panel h3 {{
      margin-top: 0;
    }}
    .panel p, .panel li, .small {{
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .dropzone {{
      border: 2px dashed rgba(15, 118, 110, 0.30);
      border-radius: 18px;
      padding: 24px;
      background: rgba(15, 118, 110, 0.06);
      text-align: center;
      transition: border-color 160ms ease, transform 160ms ease, background 160ms ease;
    }}
    .dropzone.drag {{
      border-color: var(--accent-2);
      background: rgba(194, 65, 12, 0.08);
      transform: translateY(-1px);
    }}
    .dropzone strong {{
      display: block;
      font-size: 20px;
      margin-bottom: 8px;
    }}
    .controls {{
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }}
    label {{
      font-weight: 600;
      display: block;
      margin-bottom: 6px;
    }}
    input[type="file"], textarea, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.85);
      font: inherit;
      color: var(--ink);
    }}
    textarea {{
      min-height: 92px;
      resize: vertical;
    }}
    button {{
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }}
    button:hover {{
      transform: translateY(-1px);
    }}
    button:disabled {{
      opacity: 0.55;
      cursor: progress;
      transform: none;
    }}
    .primary {{
      background: var(--accent);
      color: white;
    }}
    .secondary {{
      background: rgba(29, 78, 216, 0.10);
      color: var(--accent-3);
      border: 1px solid rgba(29, 78, 216, 0.16);
    }}
    .status {{
      min-height: 24px;
      font-weight: 600;
      color: var(--accent);
    }}
    .status.error {{
      color: #b42318;
    }}
    .status.busy {{
      color: var(--accent-2);
    }}
    .badge-row, .link-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .badge, .link-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
      text-decoration: none;
      font-size: 14px;
    }}
    .metric {{
      border-radius: 16px;
      padding: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.7);
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric .value {{
      font-size: 18px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .preview-grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
      margin-top: 18px;
    }}
    iframe {{
      width: 100%;
      min-height: 640px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: white;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(17, 24, 39, 0.96);
      color: #e5f0ff;
      padding: 16px;
      min-height: 240px;
      overflow: auto;
      font-family: "IBM Plex Mono", Consolas, monospace;
      font-size: 13px;
      line-height: 1.55;
    }}
    .artifact-list {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }}
    .artifact-item {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.76);
    }}
    .artifact-item strong {{
      display: block;
    }}
    .artifact-item span {{
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 980px) {{
      .hero-grid, .preview-grid {{
        grid-template-columns: 1fr;
      }}
      iframe {{
        min-height: 420px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">TSDataForge Local GUI</div>
      <h1>Drop one raw time-series dataset. Get a profiling report, handoff bundle, and next step.</h1>
      <p>This GUI is intentionally thin. It is a local front-end over the existing TSDataForge profiling and handoff flow, built for the exact moment when a scientist or engineer wants a result before they want an API tour.</p>
      <div class="hero-grid">
        <div class="panel">
          <h2>Bring your own file</h2>
          <div class="dropzone" id="dropzone">
            <strong>Drag a .npy, .npz, .csv, .txt, or .json file here</strong>
            <div class="small">Or browse from disk. The file stays on your machine and the bundle is written into a local output folder.</div>
          </div>
          <div class="controls">
            <div>
              <label for="fileInput">Input file</label>
              <input id="fileInput" type="file" accept=".npy,.npz,.csv,.txt,.json">
            </div>
            <div>
              <label for="goalInput">Optional goal</label>
              <textarea id="goalInput" placeholder="Example: decide whether this asset should route into anomaly detection or forecasting"></textarea>
            </div>
            <button class="primary" id="uploadButton">Build handoff bundle</button>
          </div>
        </div>
        <div class="panel">
          <h2>Or start from a built-in demo</h2>
          <p>Use this when you want the first success immediately or need a GitHub-friendly walkthrough without preparing a file.</p>
          <div class="controls">
            <div>
              <label for="scenarioSelect">Demo scenario</label>
              <select id="scenarioSelect">{scenario_options}</select>
            </div>
            <button class="secondary" id="demoButton">Run demo bundle</button>
          </div>
          <div class="badge-row">
            <span class="badge">Human first: report.html</span>
            <span class="badge">Agent first: handoff_index_min.json</span>
            <span class="badge">Default output root: .bundle/gui_runs</span>
          </div>
          <p class="small">If report generation fails, install the visualization extras: <code>pip install "tsdataforge[viz]"</code>.</p>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="metric">
        <div class="label">Status</div>
        <div class="value" id="statusText">Ready</div>
      </div>
      <div class="metric">
        <div class="label">Recommended next step</div>
        <div class="value" id="nextStepText">Waiting for a bundle</div>
      </div>
      <div class="metric">
        <div class="label">Output directory</div>
        <div class="value" id="outputDirText">Not generated yet</div>
      </div>
    </section>

    <section class="panel" style="margin-top: 18px;">
      <h2>Bundle summary</h2>
      <p id="bundleSummary">Run a demo or upload a file to see the outcome-first summary.</p>
      <div class="link-row" id="linkRow"></div>
      <div class="artifact-list" id="artifactList"></div>
    </section>

    <section class="preview-grid">
      <div class="panel">
        <h2>Report preview</h2>
        <iframe id="reportFrame" title="TSDataForge report preview"></iframe>
      </div>
      <div class="panel">
        <h3>Dataset card</h3>
        <pre id="cardPreview">No bundle yet.</pre>
        <h3 style="margin-top: 18px;">Compact context</h3>
        <pre id="contextPreview">No bundle yet.</pre>
      </div>
    </section>
  </div>

  <script>
    const scenarios = {scenarios_json};
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const goalInput = document.getElementById("goalInput");
    const uploadButton = document.getElementById("uploadButton");
    const demoButton = document.getElementById("demoButton");
    const scenarioSelect = document.getElementById("scenarioSelect");
    const statusText = document.getElementById("statusText");
    const nextStepText = document.getElementById("nextStepText");
    const outputDirText = document.getElementById("outputDirText");
    const bundleSummary = document.getElementById("bundleSummary");
    const linkRow = document.getElementById("linkRow");
    const artifactList = document.getElementById("artifactList");
    const cardPreview = document.getElementById("cardPreview");
    const contextPreview = document.getElementById("contextPreview");
    const reportFrame = document.getElementById("reportFrame");

    scenarioSelect.value = scenarios[0];

    function setBusy(message) {{
      statusText.textContent = message;
      uploadButton.disabled = true;
      demoButton.disabled = true;
    }}

    function setReady(message) {{
      statusText.textContent = message;
      uploadButton.disabled = false;
      demoButton.disabled = false;
    }}

    function setError(message) {{
      statusText.textContent = message;
      uploadButton.disabled = false;
      demoButton.disabled = false;
    }}

    async function fetchText(url) {{
      if (!url) return "";
      const response = await fetch(url);
      if (!response.ok) {{
        throw new Error("Failed to load preview: " + url);
      }}
      return await response.text();
    }}

    function renderLinks(summary) {{
      linkRow.innerHTML = "";
      const entries = [
        ["Open report", summary.links.report],
        ["Open dataset card", summary.links.card],
        ["Open compact context", summary.links.context],
        ["Open handoff index", summary.links.index_min],
        ["Open action plan", summary.links.action_plan],
        ["Open bundle JSON", summary.links.bundle],
      ];
      for (const [label, href] of entries) {{
        if (!href) continue;
        const a = document.createElement("a");
        a.className = "link-chip";
        a.href = href;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = label;
        linkRow.appendChild(a);
      }}
    }}

    function renderArtifacts(summary) {{
      artifactList.innerHTML = "";
      for (const artifact of summary.artifacts) {{
        const item = document.createElement("div");
        item.className = "artifact-item";
        const title = document.createElement(artifact.href ? "a" : "strong");
        if (artifact.href) {{
          title.href = artifact.href;
          title.target = "_blank";
          title.rel = "noopener noreferrer";
        }}
        title.textContent = artifact.name;
        item.appendChild(title);
        const meta = document.createElement("span");
        meta.textContent = artifact.kind + " - " + artifact.description;
        item.appendChild(meta);
        artifactList.appendChild(item);
      }}
    }}

    async function renderBundle(summary) {{
      bundleSummary.textContent = summary.summary || "Bundle generated.";
      nextStepText.textContent = summary.recommended_next_step || "No single next step suggested";
      outputDirText.textContent = summary.output_dir || "No output directory recorded";
      renderLinks(summary);
      renderArtifacts(summary);
      reportFrame.src = summary.links.report || "about:blank";
      cardPreview.textContent = summary.links.card ? await fetchText(summary.links.card) : "No dataset card generated.";
      contextPreview.textContent = summary.links.context ? await fetchText(summary.links.context) : "No compact context generated.";
    }}

    async function runDemo() {{
      setBusy("Running demo bundle...");
      try {{
        const response = await fetch("/api/demo", {{
          method: "POST",
          headers: {{
            "Content-Type": "application/json"
          }},
          body: JSON.stringify({{ scenario: scenarioSelect.value }})
        }});
        const payload = await response.json();
        if (!response.ok) {{
          throw new Error(payload.error || "Demo run failed");
        }}
        await renderBundle(payload);
        setReady("Demo bundle ready");
      }} catch (error) {{
        setError(error.message);
      }}
    }}

    async function runUpload() {{
      const file = fileInput.files[0];
      if (!file) {{
        setError("Choose a file first.");
        return;
      }}
      setBusy("Building handoff bundle...");
      try {{
        const form = new FormData();
        form.append("file", file);
        form.append("goal", goalInput.value || "");
        const response = await fetch("/api/handoff", {{
          method: "POST",
          body: form
        }});
        const payload = await response.json();
        if (!response.ok) {{
          throw new Error(payload.error || "Upload failed");
        }}
        await renderBundle(payload);
        setReady("Bundle ready");
      }} catch (error) {{
        setError(error.message);
      }}
    }}

    uploadButton.addEventListener("click", runUpload);
    demoButton.addEventListener("click", runDemo);

    ["dragenter", "dragover"].forEach((eventName) => {{
      dropzone.addEventListener(eventName, (event) => {{
        event.preventDefault();
        dropzone.classList.add("drag");
      }});
    }});

    ["dragleave", "drop"].forEach((eventName) => {{
      dropzone.addEventListener(eventName, (event) => {{
        event.preventDefault();
        dropzone.classList.remove("drag");
      }});
    }});

    dropzone.addEventListener("drop", (event) => {{
      const files = event.dataTransfer.files;
      if (files && files.length > 0) {{
        fileInput.files = files;
      }}
    }});
  </script>
</body>
</html>
"""


class GUIHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_cls: type[BaseHTTPRequestHandler], *, output_root: Path):
        super().__init__(server_address, handler_cls)
        self.public_host = server_address[0]
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)


class GUIRequestHandler(BaseHTTPRequestHandler):
    server: GUIHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
        return

    @property
    def origin(self) -> str:
        return f"http://{self.server.public_host}:{self.server.server_port}"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(build_gui_html())
            return
        if parsed.path == "/healthz":
            self._send_json({"ok": True, "output_root": str(self.server.output_root)})
            return
        if parsed.path.startswith("/runs/"):
            self._serve_run_file(parsed.path)
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown route.")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/demo":
            self._handle_demo()
            return
        if parsed.path == "/api/handoff":
            self._handle_handoff()
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "Unknown route.")

    def _serve_run_file(self, route_path: str) -> None:
        parts = [part for part in route_path.split("/") if part]
        if len(parts) < 3:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Missing run file path.")
            return
        run_id = parts[1]
        relative = Path(*parts[2:])
        try:
            target = _safe_child(self.server.output_root / run_id, relative)
        except PermissionError as exc:
            self._send_error_json(HTTPStatus.FORBIDDEN, str(exc))
            return
        if target.is_dir():
            index_file = target / "index.html"
            if index_file.exists():
                target = index_file
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Directory listing is not available.")
                return
        if not target.exists():
            self._send_error_json(HTTPStatus.NOT_FOUND, "File not found.")
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix.lower() == ".md":
            content_type = "text/plain; charset=utf-8"
        elif target.suffix.lower() == ".json":
            content_type = "application/json; charset=utf-8"
        elif target.suffix.lower() == ".html":
            content_type = "text/html; charset=utf-8"

        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_demo(self) -> None:
        payload = self._read_json_body()
        scenario = str(payload.get("scenario") or DEMO_SCENARIOS[0])
        if scenario not in DEMO_SCENARIOS:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Unsupported demo scenario: {scenario}")
            return

        run_id = _run_id(f"demo-{scenario}")
        output_dir = self.server.output_root / run_id
        try:
            from .surface import demo

            bundle = demo(output_dir=output_dir, scenario=scenario)
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._send_json(bundle_summary_for_gui(bundle, run_id=run_id, origin=self.origin, input_name=scenario))

    def _handle_handoff(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Expected multipart form upload.")
            return

        raw_length = self.headers.get("Content-Length", "0")
        length = int(raw_length) if raw_length.isdigit() else 0
        fields = _parse_multipart_form(content_type, self.rfile.read(length))
        file_item = fields.get("file")
        if file_item is None or not file_item.get("filename"):
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing uploaded file.")
            return

        goal = fields.get("goal", {}).get("value")
        filename = _slugify_filename(Path(str(file_item["filename"])).name)
        run_id = _run_id(Path(filename).stem or "upload")
        output_dir = self.server.output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        upload_path = output_dir / filename
        upload_path.write_bytes(bytes(file_item["data"]))

        try:
            from .surface import handoff

            bundle = handoff(upload_path, output_dir=output_dir, goal=str(goal) if goal else None)
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._send_json(bundle_summary_for_gui(bundle, run_id=run_id, origin=self.origin, input_name=filename))

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        length = int(raw_length) if raw_length.isdigit() else 0
        if length <= 0:
            return {}
        payload = self.rfile.read(length)
        if not payload:
            return {}
        return json.loads(payload.decode("utf-8"))

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


@dataclass
class GUIApp:
    server: GUIHTTPServer
    url: str
    output_root: Path

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def create_gui_app(
    *,
    host: str = DEFAULT_GUI_HOST,
    port: int = DEFAULT_GUI_PORT,
    output_root: str | Path = DEFAULT_GUI_OUTPUT_ROOT,
) -> GUIApp:
    root = Path(output_root)
    server = GUIHTTPServer((host, int(port)), GUIRequestHandler, output_root=root)
    url = f"http://{server.public_host}:{server.server_port}"
    return GUIApp(server=server, url=url, output_root=root.resolve())


def launch_gui(
    *,
    host: str = DEFAULT_GUI_HOST,
    port: int = DEFAULT_GUI_PORT,
    output_root: str | Path = DEFAULT_GUI_OUTPUT_ROOT,
    open_browser: bool = True,
) -> GUIApp:
    app = create_gui_app(host=host, port=port, output_root=output_root)
    print(f"TSDataForge GUI running at {app.url}")
    print(f"Output root: {app.output_root}")
    print("Press Ctrl+C to stop the local server.")
    if open_browser:
        webbrowser.open(app.url)
    try:
        app.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        app.shutdown()
    return app


__all__ = [
    "DEFAULT_GUI_HOST",
    "DEFAULT_GUI_PORT",
    "DEFAULT_GUI_OUTPUT_ROOT",
    "DEMO_SCENARIOS",
    "GUIApp",
    "build_gui_html",
    "bundle_summary_for_gui",
    "create_gui_app",
    "launch_gui",
]
