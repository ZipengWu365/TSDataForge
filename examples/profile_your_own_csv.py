import pandas as pd

from tsdataforge import handoff, report


df = pd.read_csv("my_series.csv")
values = df[["sensor_a", "sensor_b"]].to_numpy()
time = df["timestamp"].to_numpy()

report(
    values,
    time=time,
    output_path="report.html",
    dataset_id="my_series",
    channel_names=["sensor_a", "sensor_b"],
)

bundle = handoff(
    values,
    time=time,
    output_dir="my_series_handoff",
    dataset_id="my_series",
    channel_names=["sensor_a", "sensor_b"],
)
print(bundle.output_dir)
