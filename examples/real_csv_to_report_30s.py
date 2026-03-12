import numpy as np
from tsdataforge import report

# Direct CSV loading expects a plain numeric matrix.
# If the first column is monotonic increasing, TSDataForge treats it as time.
values = np.random.default_rng(0).normal(size=(12, 128))
np.savetxt("demo.csv", values, delimiter=",")
report("demo.csv", output_path="demo_report.html")
