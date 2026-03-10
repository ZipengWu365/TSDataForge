import numpy as np
from tsdataforge import handoff

v1 = np.random.default_rng(0).normal(size=(8, 96))
v2 = v1.copy()
v2[:, 60:] += 0.5

handoff(v1, output_dir="bundle_v1")
handoff(v2, output_dir="bundle_v2")
print("Compare bundle_v1 and bundle_v2 using their cards, contexts, and reports.")
