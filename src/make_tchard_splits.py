"""
Build epitope-disjoint (hard) splits on TChard, plus a random split control.

No epitope in test appears in train or val. Same for val vs train.

Input: TChard ds.csv (or the zip)
Output: splits/tchard/<neg-mode>/<split-type>/seed{i}/{train,val,test}.csv & manifest.json

To run it, use: python src/make_tchard_splits.py --input data/raw/tc-hard.zip --neg-mode RN
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
NOT_AA = f"[^{AA}]"

CANON = {"antigen.epitope": "peptide", "cdr3.beta": "B3", "label": "binder"}

# NA is almost entirely nettcr-2.0 (10x).
# iedb and mira are under 0.5% of negatives
RN_SOURCES = {"randomized"}
NA_SOURCES = {"nettcr-2.0", "iedb", "mira"}

OUT_COLUMNS = ["row_id", "peptide", "B3", "binder"]


READ_OPTS = dict(
    low_memory=False,
    keep_default_na=False,
    na_values=["", "NaN", "nan", "NULL", "null", "None"],
)

MAXLEN_PEP = 16
MINLEN_B3 = 9
MAXLEN_B3 = 23
TEST_RATIO = 0.15
VAL_RATIO = 0.15


def load_tchard(path):
    """
    Read TChard and return peptide/ B3/ binder/ negative.source.
    """

    path = Path(path)

    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        members = []
        for item in archive.infolist():
            if item.filename.lower().endswith(".csv") and not item.filename.startswith("__MACOSX"):
                members.append(item)

        if len(members) == 0:
            raise SystemExit(f"No CSV inside {path}")

        biggest = max(members, key=lambda m: m.file_size)
        print(f"Reading {biggest.filename} from {path.name}")
        df = pd.read_csv(archive.open(biggest), **READ_OPTS)
    else:
        df = pd.read_csv(path, **READ_OPTS)

    missing = []
    for column in CANON:
        if column not in df.columns:
            missing.append(column)

    if missing:
        raise SystemExit(f"These columns are missing: {missing}\n")

    data = pd.DataFrame()
    data["peptide"] = df["antigen.epitope"].astype(str).str.strip().str.upper()
    data["B3"] = df["cdr3.beta"].astype(str).str.strip().str.upper()
    data["binder"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    data["negative.source"] = df["negative.source"]
    if "license" in df.columns:
        data["license"] = df["license"]

    print(f"Read {len(data):,} rows, {data['peptide'].nunique():,} epitopes, "
          f"{data['binder'].sum():,} positives ({data['binder'].mean():.1%})")
    return data


def filter_negatives(data, mode):
    """
    Keep all positives plus the negatives belonging to one scheme (RN/ NA/ full).
    """

    if mode == "full":
        print(f"filter_negatives(full): {len(data):,} rows")
        return data

    sources = {"RN": RN_SOURCES, "NA": NA_SOURCES}[mode]
    # positives have no negative.source, so filtering on that column alone would drop all 150k of them
    is_positive = data["binder"] == 1
    keep = is_positive | data["negative.source"].isin(sources)

    out = data[keep].reset_index(drop=True)
    n_pos = int(out["binder"].sum())
    print(f"filter_negatives({mode}): {len(data):,} -> {len(out):,} rows "
          f"({n_pos:,} pos, {len(out) - n_pos:,} neg)")
    return out


def drop_bad_residues(data):
    """
    Drop rows whose peptide or B3 has a non-standard residue.
    """

    bad = (data["peptide"].str.contains(NOT_AA, regex=True)
           | data["B3"].str.contains(NOT_AA, regex=True))

    print(f"drop_bad_residues: dropping {int(bad.sum()):,} rows")
    return data[~bad].reset_index(drop=True)


def deduplicate(data):
    """
    Remove exact (peptide, B3, binder) repeats.

    Two receptors differing only in their alpha chain become the same input once
    we keep CDR3-beta only. The paper de-duplicates on these same three columns.
    """

    before = len(data)
    out = data.drop_duplicates(subset=["peptide", "B3", "binder"]).reset_index(drop=True)

    print(f"deduplicate: {before:,} -> {len(out):,} rows (dropped {before - len(out):,})")
    return out


def resolve_label_conflicts(data, policy="drop"):
    """
    Handle (peptide, B3) pairs labelled both binding and non-binding.
    """

    n_labels = data.groupby(["peptide", "B3"])["binder"].transform("nunique")
    conflicted = n_labels > 1
    n_pairs = len(data[conflicted][["peptide", "B3"]].drop_duplicates())

    if policy == "drop":
        out = data[~conflicted]
    elif policy == "positive":
        out = data[~conflicted | (data["binder"] == 1)]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    out = out.reset_index(drop=True)
    print(f"resolve_label_conflicts({policy}): {n_pairs:,} conflicting pairs, "
          f"{len(data):,} -> {len(out):,} rows")
    return out


def filter_lengths(data):
    """
    Apply the paper's length window and print the real maximum lengths.
    """

    pep_len = data["peptide"].str.len()
    b3_len = data["B3"].str.len()
    keep = (pep_len <= MAXLEN_PEP) & (b3_len >= MINLEN_B3) & (b3_len <= MAXLEN_B3)

    out = data[keep].reset_index(drop=True)

    print(f"filter_lengths: {len(data):,} -> {len(out):,} rows")
    # these are the maxlen values the cnn yaml needs
    print(f"  peptide max {out['peptide'].str.len().max()}, "
          f"B3 max {out['B3'].str.len().max()}")
    return out


def assign_row_id(data):
    """
    Add a globally unique row_id. Run after all cleaning, before any splitting.
    """

    out = data.reset_index(drop=True).copy()
    out.insert(0, "row_id", np.arange(len(out)))
    return out


def epitope_disjoint_split(data, min_ratio, low, high, rng):
    budget = round(min_ratio * len(data))
    counts = data.groupby("peptide").size()
    candidates = np.asarray(counts[(counts >= low) & (counts <= high)].index)

    held_epitopes = []
    held_rows = 0

    for i in rng.permutation(len(candidates)):
        if held_rows >= budget:
            break
        epitope = str(candidates[i])
        held_epitopes.append(epitope)
        held_rows += int(counts[epitope])

    if held_rows < budget:
        raise ValueError(
            f"Ran out of candidates. Took all {len(candidates)} eligible epitopes "
            f"({held_rows} rows) but the budget is {budget} of {len(data)} rows"
        )

    taken = data["peptide"].isin(held_epitopes)
    rest = data[~taken].reset_index(drop=True)
    held = data[taken].reset_index(drop=True)

    return rest, held, held_epitopes


def make_three_way(data, test_ratio, val_ratio, low, high, val_low, val_high, seed):
    """
    Cut test, then val out of what is left, with one seed driving both.
    """

    rng = np.random.default_rng(seed)
    pool, test, test_epitopes = epitope_disjoint_split(data, test_ratio, low, high, rng)
    train, val, val_epitopes = epitope_disjoint_split(pool, val_ratio, val_low, val_high, rng)

    return {
        "train": train,
        "val": val,
        "test": test,
        "test_epitopes": sorted(test_epitopes),
        "val_epitopes": sorted(val_epitopes),
    }


def random_split(data, test_ratio, val_ratio, seed):
    """
    Split by row, ignoring epitopes. The control condition.
    """

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))

    n_test = round(test_ratio * len(data))
    n_val = round(val_ratio * len(data))

    return {
        "train": data.iloc[order[n_test + n_val:]].reset_index(drop=True),
        "val": data.iloc[order[n_test:n_test + n_val]].reset_index(drop=True),
        "test": data.iloc[order[:n_test]].reset_index(drop=True),
        "test_epitopes": [],
        "val_epitopes": [],
    }


def describe(splits):
    """
    Rows, positives, epitopes and B3 counts per split. Returns the table.
    """

    summary = []

    for split_name in ["train", "val", "test"]:
        rows = splits[split_name]
        summary.append({
            "split": split_name,
            "n_rows": len(rows),
            "n_pos": int(rows["binder"].sum()),
            "pos_frac": round(float(rows["binder"].mean()), 4) if len(rows) else None,
            "n_epitopes": rows["peptide"].nunique(),
            "n_b3": rows["B3"].nunique(),
        })

    table = pd.DataFrame(summary)
    print(table.to_string(index=False))
    return table


def check_leakage(splits, split_type):
    """
    Assert the splits are disjoint in the way the split type claims.
    """

    row_ids = []
    for split_name in ["train", "val", "test"]:
        row_ids.append(set(splits[split_name]["row_id"]))

    assert not (row_ids[0] & row_ids[1]), "row_id overlap between train and val"
    assert not (row_ids[0] & row_ids[2]), "row_id overlap between train and test"
    assert not (row_ids[1] & row_ids[2]), "row_id overlap between val and test"

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared_epitopes = set(splits[a]["peptide"]) & set(splits[b]["peptide"])
        shared_b3 = set(splits[a]["B3"]) & set(splits[b]["B3"])
        print(f"  {a} and {b}: {len(shared_epitopes)} shared epitopes, "
              f"{len(shared_b3)} shared B3")

        if split_type == "hard":
            assert not shared_epitopes, (
                f"Epitope leak between {a} and {b}: {sorted(shared_epitopes)[:5]}"
            )


def per_epitope_counts(data):
    """
    Row, positive and negative counts per epitope. Same grouping as per_epitope() in evaluation.py, so the two tables can be merged.
    """

    counts = data.groupby("peptide")["binder"].agg(n="size", n_pos="sum").reset_index()
    counts["n_neg"] = counts["n"] - counts["n_pos"]
    return counts.sort_values("n", ascending=False)


def write_splits(splits, out_dir):
    """
    Write the three splits, a copy keeping negative.source, and the test counts.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        rows = splits[split_name]
        rows[OUT_COLUMNS].to_csv(out_dir / f"{split_name}.csv", index=False)
        rows.to_csv(out_dir / f"{split_name}_full.csv", index=False)

    per_epitope_counts(splits["test"]).to_csv(out_dir / "test_per_epitope.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Create the TChard hard splits")
    parser.add_argument("--input", required=True, help="path to tc-hard.zip or ds.csv")
    parser.add_argument("--neg-mode", choices=["RN", "NA", "full"], default="RN")
    parser.add_argument("--split-type", choices=["hard", "random"], default="hard")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--low", type=int, default=500, help="min rows per candidate epitope")
    parser.add_argument("--high", type=int, default=10000, help="max rows per candidate epitope")
    args = parser.parse_args()

    data = load_tchard(args.input)
    data = filter_negatives(data, args.neg_mode)
    data = drop_bad_residues(data)
    data = deduplicate(data)
    data = resolve_label_conflicts(data)
    data = filter_lengths(data)
    data = assign_row_id(data)
    print(f"\nClean dataset: {len(data):,} rows, {data['peptide'].nunique():,} epitopes\n")

    # the pool is smaller once test is gone, so val needs a lower floor
    val_low = args.low // 2

    out_dir = Path("splits/tchard") / args.neg_mode / args.split_type
    manifest = {
        "params": {
            "neg_mode": args.neg_mode,
            "split_type": args.split_type,
            "low": args.low,
            "high": args.high,
            "val_low": val_low,
            "n_rows": len(data),
            "n_epitopes": int(data["peptide"].nunique()),
        },
        "seeds": {},
    }

    for seed in args.seeds:
        print(f"seed {seed}")

        if args.split_type == "hard":
            splits = make_three_way(data, TEST_RATIO, VAL_RATIO,
                                    args.low, args.high, val_low, args.high, seed)
        else:
            splits = random_split(data, TEST_RATIO, VAL_RATIO, seed)

        table = describe(splits)
        check_leakage(splits, args.split_type)
        write_splits(splits, out_dir / f"seed{seed}")

        manifest["seeds"][seed] = {
            "stats": table.to_dict(orient="records"),
            "test_epitopes": splits["test_epitopes"],
            "val_epitopes": splits["val_epitopes"],
            "test_frac": round(len(splits["test"]) / len(data), 4),
            "val_frac": round(len(splits["val"]) / len(data), 4),
        }
        print()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"The splits and manifest are written to {out_dir}")

if __name__ == "__main__":
    main()