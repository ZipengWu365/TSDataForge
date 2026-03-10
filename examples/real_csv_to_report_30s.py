import numpy as np
from tsdataforge import report

values = np.random.default_rng(0).normal(size=(12, 128))
np.savetxt("demo.csv", values, delimiter=",")
report("demo.csv", output_path="demo_report.html")
