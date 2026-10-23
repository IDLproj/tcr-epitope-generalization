"""Convert NetTCR's own cv predictions into our format, as a reference point.

The authors already provide their cross-validation predictions, so we don’t need to run inference ourselves. However, they use all six CDRs instead of only B3, so we can use the results as a reference but not as a baseline.
"""

import argparse
import glob
from pathlib import Path
import pandas as pd

from convert_nettcr import from_nettcr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pan", "pretrained"], default="pretrained")
    ap.add_argument("--out", type=Path, default=Path("preds"))
    args = ap.parse_args()

    base = "external/NetTCR-2.2/models"
    if args.mode == "pan":
        raw = pd.read_csv(f"{base}/nettcr_2_2_pan/cv_pred_df.csv")
    else:
        files = sorted(glob.glob(f"{base}/nettcr_2_2_pretrained/*/cv_pred_df.csv"))
        raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    out = from_nettcr(raw)
    out["setting"] = "full"      # use all six CDRs
    out["model"] = f"nettcr_published_{args.mode}"

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"nettcr_published_{args.mode}.csv"
    out.to_csv(args.out / name, index=False)
    print(f"Wrote {args.out / name}")


if __name__ == "__main__":
    main()