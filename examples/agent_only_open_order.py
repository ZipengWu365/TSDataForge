import json
from pathlib import Path

from tsdataforge import demo

bundle = demo(output_dir="agent_demo_bundle", scenario="ecg_public")
index = json.loads(Path("agent_demo_bundle/handoff_index_min.json").read_text(encoding="utf-8"))
print(index["agent_open_order"])
print(index["recommended_next_step"])
