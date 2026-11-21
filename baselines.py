"""Step 0: constant baselines scored on train with the official mIoU, and one valid test submission.
    python baselines.py
Writes results/step0_baselines.csv (per image) and results/sub_step0_allzeros.pt."""
import numpy as np
import pandas as pd

import data
import metrics


def main():
    df = data.train_table()
    gts = data.train_masks128()
    zeros, ones = np.zeros((128, 128), np.uint8), np.ones((128, 128), np.uint8)
    rows = []
    for name in ["all-0", "all-1", "oracle regime (not submittable)"]:
        for r in df.itertuples():
            if name == "all-0": p = zeros
            elif name == "all-1": p = ones
            else: p = ones if r.full_watermark else zeros      # uses the train label: upper bound of a regime classifier
            rows.append(dict(baseline=name, name=r.name, method=r.method, regime=r.regime,
                             miou=metrics.miou_image(p, gts[r.name])))
    res = pd.DataFrame(rows)
    data.RESULTS.mkdir(exist_ok=True)
    res.to_csv(data.RESULTS / "step0_baselines.csv", index=False)
    for b, g in res.groupby("baseline", sort=False):
        print(f"\n== {b}\n{metrics.breakdown(g)}")

    path = metrics.write_submission({n: zeros for n in data.test_names()}, data.RESULTS / "sub_step0_allzeros.pt")
    print("\nwrote", path, f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
