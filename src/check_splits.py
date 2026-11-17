from pathlib import Path
import pandas as pd

"""
Check the splits made by make_tchard_splits.py.

Run from the repo root:
    python src/check_splits.py
To save the output:
    python src/check_splits.py > reports/split_checks.txt

This file checks four things:
  1. the rows are not abmigious 
  2. which negative scheme each split holds
  3. the splits really are epitope-disjoint
  4. how many test epitopes we get
"""

NOT_AA = "[^ACDEFGHIKLMNPQRSTVWY]"
SPLIT_NAMES = ["train", "val", "test"]
MAXLEN_PEP = 16
MINLEN_B3 = 9
MAXLEN_B3 = 23


def row(label, value):
    print("  " + str(label).ljust(32) + str(value))

# Loading splits from CSV files
def load(seed_dir, name):
    full = seed_dir / (name + "_full.csv")
    if full.exists():
        return pd.read_csv(full, low_memory=False)
    return pd.read_csv(seed_dir / (name + ".csv"), low_memory=False)

def load_all(seed_dir):
    splits = {}
    for name in SPLIT_NAMES:
        splits[name] = load(seed_dir, name)
    return splits

def seed_dirs_in(folder):
    found = []
    for child in sorted(folder.iterdir()):
        if child.is_dir() and child.name.startswith("seed"):
            found.append(child)
    return found

# Data checks
def check_cleaning(splits):
    print("\nCLEANING (all should be 0)")
    data = pd.concat(splits.values(), ignore_index=True)
    bad_peptide = data["peptide"].str.contains(NOT_AA, regex=True)
    bad_b3 = data["B3"].str.contains(NOT_AA, regex=True)
    row("rows with bad residues", int((bad_peptide | bad_b3).sum()))
    labels_per_pair = data.groupby(["peptide", "B3"])["binder"].nunique()
    row("pairs with conflicting labels", int((labels_per_pair > 1).sum()))
    distinct = len(data[["peptide", "B3", "binder"]].drop_duplicates())
    row("duplicate rows", len(data) - distinct)
    too_long = data["peptide"].str.len() > MAXLEN_PEP
    b3_len = data["B3"].str.len()
    bad_b3_len = (b3_len < MINLEN_B3) | (b3_len > MAXLEN_B3)
    row("rows outside length limits", int((too_long | bad_b3_len).sum()))
    row("repeated row_id", len(data) - data["row_id"].nunique())

def check_negatives(splits):
    print("\nNEGATIVES AND POSITIVES PER SPLIT")
    for name in SPLIT_NAMES:
        data = splits[name]
        negatives = data[data["binder"] == 0]

        text = (f"{int(data['binder'].sum()):,}" + " pos (" + f"{data['binder'].mean():.0%}" + "), " + f"{len(negatives):,}" + " neg")

        if "negative.source" in data.columns and len(negatives) > 0:
            counts = negatives["negative.source"].value_counts()
            for source in counts.index:
                text = (text + "  " + str(source) + " " + f"{counts[source] / len(negatives):.0%}")

        row(name, text)

def check_overlap(splits, split_type):
    print("\nOVERLAP WITH TRAIN")

    train_epitopes = set(splits["train"]["peptide"])
    train_b3 = set(splits["train"]["B3"])

    for name in ["val", "test"]:
        epitopes = set(splits[name]["peptide"])
        b3 = set(splits[name]["B3"])
        shared_epitopes = len(train_epitopes & epitopes)
        shared_b3 = len(train_b3 & b3)

        row(name, str(shared_epitopes) + " of " + str(len(epitopes))
            + " epitopes shared, " + f"{shared_b3 / len(b3):.0%}" + " of B3 shared")

    if split_type == "hard" and len(train_epitopes & set(splits["test"]["peptide"])) > 0:
        print("  EPITOPE LEAK - the hard split is not disjoint")

def check_seeds(seed_dirs):
    print("\nTEST EPITOPES PER SEED")

    everything = set()
    total = 0
    for seed_dir in seed_dirs:
        epitopes = set(load(seed_dir, "test")["peptide"])
        everything = everything | epitopes
        total = total + len(epitopes)
        row(seed_dir.name, len(epitopes))
    row("total points for the plot", total)
    row("distinct epitopes used", len(everything))


def compare_modes(root):
    print("\nCHECK IF THE SPLITS HOLD THE SAME EPITOPES")
    test_sets = {}
    for mode in ["RN", "NA", "full"]:
        folder = root / mode / "hard"
        if folder.exists() and len(seed_dirs_in(folder)) > 0:
            seed0 = seed_dirs_in(folder)[0]
            test_sets[mode] = set(load(seed0, "test")["peptide"])

    modes = sorted(test_sets.keys())
    if len(modes) < 2:
        print("  need at least two modes generated to compare")
        return

    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a = test_sets[modes[i]]
            b = test_sets[modes[j]]
            row(modes[i] + " vs " + modes[j],
                str(len(a & b)) + " shared of " + str(len(a | b)))

    print("\n  Low overlap means an RN vs NA score gap is partly a")
    print("  different test set, not only different negatives.")


def main():
    root = Path("splits/tchard")
    if not root.exists():
        raise SystemExit("No splits found at " + str(root))

    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue

        for type_dir in sorted(mode_dir.iterdir()):
            if not type_dir.is_dir():
                continue

            seed_dirs = seed_dirs_in(type_dir)
            if len(seed_dirs) == 0:
                continue

            print("\n" + "=" * 60)
            print(str(type_dir))
            print("=" * 60)

            splits = load_all(seed_dirs[0])
            check_cleaning(splits)
            check_negatives(splits)
            check_overlap(splits, type_dir.name)
            check_seeds(seed_dirs)

    print("\n" + "=" * 60)
    compare_modes(root)


if __name__ == "__main__":
    main()