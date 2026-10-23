"""Convert between our schema and NetTCR-2.2's input / output format.

NetTCR takes 6 sequences as input (A1, A2, A3, B1, B2, B3).
TChard only has B3, so we fill other five with empty strings. NetTCR pad the them to -5, which means that the model see those channels as empty.

This only works if we retrain. The publish weights were trained on real alpha-chain data and would give meaningless scores on all-padding input.
"""

import argparse
from pathlib import Path 

import pandas as pd

def to_nettcr(df):
    """ Conver our split data to NetTCR's input format. """

    out = pd.DataFrame()

    out["index"] = df["row_id"]
    out["A1"] = ""
    out["A2"] = ""
    out["A3"] = ""
    out["B1"] = ""
    out["B2"] = ""
    out["B3"] = df["B3"]
    out["peptide"] = df["peptide"]
    out["binder"] = df["binder"]
    out["partition"] = df["partition"]

    return out

def from_nettcr(pred_df):
    """ Takes the NetTCR output as input and convert it to our evaluation format."""

    out = pd.DataFrame()

    out["row_id"] = pred_df["index"]
    out["peptide"] = pred_df["peptide"]
    out["B3"] = pred_df["B3"]
    out["binder"] = pred_df["binder"]
    out["pred"] = pred_df["prediction"]
    out["model"] = "nettcr"
    out["setting"] = "B3"

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type = Path, help = "input CSV")
    ap.add_argument("--out", type = Path, default=Path("results"))
    ap.add_argument("--direction", choices=["to", "from"], default="to")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.direction == "to":
        output_csv = to_nettcr(df)
        name = "nettcr_input.csv"
    elif args.direction == "from":
        output_csv = from_nettcr(df)
        name = "nettcr_preds.csv"

    args.out.mkdir(parents= True, exist_ok=True)
    output_csv.to_csv(args.out / name, index = False)
    print(f"Wrote {args.out / name}")

if __name__ == "__main__":
    main()