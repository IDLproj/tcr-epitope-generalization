"""
Evaluation for all three models. 

Input: predictions (csv)
Output: results/summary.csv (per model and setting) & results/per_epitope.csv

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

REQUIRED = ["row_id", "peptide", "B3", "binder", "pred", "model", "setting"]

def validate(df):
    """
    Check the predictions CSV has the right columns and valid values.
    """

    missing = []
    for c in REQUIRED:
        if c not in df.columns:
            missing.append(c)
    
    if len(missing) > 0:
        raise ValueError(f"Missing Column {missing}")
    
    if len(np.unique(df["row_id"])) < len(df["row_id"]):
        raise ValueError("row_id repeats")

    if np.max(df["pred"]) > 1 or np.min(df["pred"]) < 0:
        raise ValueError(f"The predicted value is not in 0 - 1. Min = {df['pred'].min()}, Max = {df['pred'].max()}")
    
    if not df["binder"].isin([0, 1]).all():
        raise ValueError("The binder value not 0 or 1.")

    

def evaluate_metrics(y_true, y_pred):
    """
    Takes (y_true, y_pred) and return a dict with auc/ auc01/ aupr.
    """

    # in case the binders are all 0 or all 1
    if len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "auc01": np.nan, "aupr": np.nan}
    
    return {
        "auc": roc_auc_score(y_true, y_pred), 
        "auc01": roc_auc_score(y_true, y_pred, max_fpr=0.1),
        "aupr":average_precision_score(y_true, y_pred)
    }

def per_epitope(df):
    """
    Metrics for each peptide. return a DataFrame and one row per epitope.
    """

    epitope_list = []

    for peptide, g in df.groupby("peptide"):
        m = evaluate_metrics(g["binder"].values, g["pred"].values) # get auc, auc01, aupr
        # .values makes sure that is np.array (not pandas series)
        m["n_pos"] = g["binder"].sum()
        m["peptide"] = peptide
        m["n"] = len(g)

        epitope_list.append(m)

    return pd.DataFrame(epitope_list)



def evaluate(df):
    """
    Metrics for each (model, setting). 
    Returns (summary, per_epitope) DataFrames.
    """
    evaluate_finalll = [] # summary of rows
    frames = []

    for (model, setting), g in df.groupby(["model", "setting"]):
        m = evaluate_metrics(g["binder"].values, g["pred"].values) # get auc, auc01, aupr     
        m["model"] = model
        m["setting"] = setting
        m["n_pos"] = g["binder"].sum()
        m["n_epitopes"] = g["peptide"].nunique()
        m["n"] = len(g)

        evaluate_finalll.append(m)
        
        # get per_epitope
        pe = per_epitope(g)
        pe["model"] = model
        pe["setting"] = setting

        frames.append(pe)

    return pd.DataFrame(evaluate_finalll), pd.concat(frames, ignore_index= True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type = Path, help = "input CSV")
    ap.add_argument("--out", type = Path, default=Path("results"))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    validate(df)
    summary, per_epitope_df = evaluate(df)

    args.out.mkdir(parents= True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index = False)
    per_epitope_df.to_csv(args.out / "per_epitope.csv", index = False)
    print(summary)

if __name__ == "__main__":
    main()