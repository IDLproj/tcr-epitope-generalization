from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from baseline_lr import SETTINGS, coefficient_table, fit_and_predict
from evaluation import evaluate_metrics
from sklearn.metrics import roc_auc_score, average_precision_score
"""
Notes about this file:
This file runs the logistic-regression baselines.

This is how to run it: 
# This is with 3 settings (i.e. concat, tcr_only, interaction)
python src/run_lr.py   
# This is with 1 setting 
python src/run_lr.py --settings concat 
"""
SPLIT_DIR = Path("splits")
PRED_DIR = Path("results/predictions_lr")

def main() -> None:
    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default=str(SPLIT_DIR))
    ap.add_argument("--out-dir", default=str(PRED_DIR))
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS))
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--coefs", action="store_true", help="print top coefficients")
    args = ap.parse_args()

    # Load the train and test splits
    split_dir = Path(args.split_dir)
    train = pd.read_csv(split_dir / "train.csv")
    test = pd.read_csv(split_dir / "test.csv")
    print(f"Train has:{len(train)} rows\n")
    print(f"Test has:{len(test)} rows\n")

    # Create the output directory 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run the logistic regression model for each setting
    for setting in args.settings:
        # Fit the model and make predictions
        preds = fit_and_predict(train, test, setting=setting, C=args.C)
        # Write the predictions to a CSV file
        path = out_dir / f"preds_lr_{setting}.csv"
        preds.to_csv(path, index=False)

        # Evaluate the predictions and print the metrics
        auroc = roc_auc_score(preds["binder"], preds["pred"])
        auc01 = roc_auc_score(preds["binder"], preds["pred"], max_fpr=0.1)
        auprc = average_precision_score(preds["binder"], preds["pred"])
        print("Setting metrics:" + setting + "\n")
        print("auc   =", round(auroc, 4))
        print("auc01 =", round(auc01, 4))
        print("aupr  =", round(auprc, 4))

        # Print the top coefficients for the model
        if args.coefs:
            print(coefficient_table(train, setting=setting, C=args.C).to_string(index=False)+ "\n")


if __name__ == "__main__":
    main()