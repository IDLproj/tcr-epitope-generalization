import argparse
import zipfile
from pathlib import Path
import pandas as pd

"""
Notes:
Inspect the TChard dataset and write a short report.

Run from the repo root:
    python src/inspect_tchard.py --input data/raw/tc-hard.zip

If a detected column is wrong, override it:
    python src/inspect_tchard.py --input data/raw/tc-hard.zip --b3-col cdr3_beta
"""
AA = set("ACDEFGHIKLMNPQRSTVWY")
MAXLEN_B3 = 23
MAXLEN_PEP = 16

# TChard marks assay-derived negatives with the literal string "NA".
# pandas reads this as a missing value by default, which would erase the
# difference between the two negative schemes. So name the nulls ourselves.
READ_OPTS = dict(
    low_memory=False,
    keep_default_na=False,
    na_values=["", "NaN", "nan", "NULL", "null", "None"],
)

OUT = []

def say(line=""):
    print(line)
    OUT.append(line)

def row(label, value):
    say("  " + str(label).ljust(32) + str(value))


# Loading
def load(path):
    if path.suffix != ".zip":
        return pd.read_csv(path, **READ_OPTS)

    archive = zipfile.ZipFile(path)

    biggest = None
    for item in archive.infolist():
        if not item.filename.lower().endswith(".csv"):
            continue
        if item.filename.startswith("__MACOSX"):
            continue
        if biggest is None or item.file_size > biggest.file_size:
            biggest = item

    if biggest is None:
        raise SystemExit("No CSV file found inside " + str(path))

    say("Reading " + biggest.filename + " (" + f"{biggest.file_size / 1e6:.0f}" + " MB uncompressed)")

    return pd.read_csv(archive.open(biggest), **READ_OPTS)

# Working out the columns
def is_sequence(series):
    sample = series.dropna().astype(str).head(2000)
    if len(sample) == 0:
        return False

    good = 0
    for value in sample:
        if len(value) >= 5 and set(value.upper()) <= AA:
            good = good + 1

    return good / len(sample) > 0.8

def find_sequence_columns(df):
    found = []
    for column in df.columns:
        if is_sequence(df[column]):
            found.append(column)
    return found

def find_peptide_column(sequence_columns):
    for column in sequence_columns:
        name = column.lower()
        if "pep" in name or "epitope" in name or "anti" in name:
            return column
    return None

def find_b3_column(sequence_columns):
    for column in sequence_columns:
        name = column.lower()
        flat = name.replace("_", "").replace(".", "")
        if "b3" in name or "cdr3b" in flat or "beta" in name:
            return column
    return None

def find_binder_column(df):
    for column in df.columns:
        name = column.lower()
        if name in ("binder", "label", "y", "target", "binding"):
            return column
        if "label" in name or "binder" in name:
            return column
    return None

def find_negative_source_column(df):
    for column in df.columns:
        name = column.lower()
        if "negative" in name or ("neg" in name and "source" in name):
            return column

    for column in df.columns:
        values = set(df[column].dropna().astype(str).unique())
        if "NA" in values and "RN" in values:
            return column

    return None

def mark_positives(series, wanted):
    try:
        return series.astype(float) == float(wanted)
    except (ValueError, TypeError):
        return series.astype(str) == str(wanted)

def shortest_and_longest(df, sequence_columns):
    medians = []
    for i in range(len(sequence_columns)):
        column = sequence_columns[i]
        median = df[column].dropna().astype(str).str.len().median()
        medians.append((median, i, column))
    medians.sort()

    return medians[0][2], medians[-1][2]

def detect_columns(df, args):
    sequence_columns = find_sequence_columns(df)

    peptide = args.peptide_col
    if not peptide:
        peptide = find_peptide_column(sequence_columns)

    b3 = args.b3_col
    if not b3:
        b3 = find_b3_column(sequence_columns)

    if (not peptide or not b3) and len(sequence_columns) >= 2:
        shortest, longest = shortest_and_longest(df, sequence_columns)
        if not peptide:
            peptide = shortest
        if not b3:
            b3 = longest

    binder = args.binder_col
    if not binder:
        binder = find_binder_column(df)

    negative_source = args.neg_source_col
    if not negative_source:
        negative_source = find_negative_source_column(df)

    return peptide, b3, binder, negative_source

def find_bad_residues(series):
    bad_rows = 0
    bad_characters = set()

    for value in series.astype(str):
        extra = set(value.upper()) - AA
        if len(extra) > 0:
            bad_rows = bad_rows + 1
            for character in extra:
                bad_characters.add(character)

    return bad_rows, ", ".join(sorted(bad_characters))

# Report sections
def report_mapping(peptide, b3, binder, negative_source):
    say("\nCOLUMN MAPPING  (maps TChard columns to expected names)")
    row("peptide", peptide)
    row("B3", b3)
    row("binder", binder)
    row("negative scheme", negative_source or "NOT FOUND")

def report_summary(work, peptide, b3, is_positive):
    n_positive = int(is_positive.sum())
    n_negative = int((~is_positive).sum())

    say("\nSUMMARY")
    row("distinct (peptide, B3) pairs", f"{len(work[[peptide, b3]].drop_duplicates()):,}")
    row("unique epitopes", f"{work[peptide].nunique():,}")
    row("unique B3 sequences", f"{work[b3].nunique():,}")
    row("positive rows (Binders)", f"{n_positive:,}" + "  (" + f"{n_positive / len(work):.1%}" + ")")
    row("negative rows", f"{n_negative:,}" + "  (" + f"{n_negative / len(work):.1%}" + ")")
    row("negative to positive ratio", f"{n_negative / n_positive:.2f}" + " : 1")

def report_negative_types(work, negative_source, is_positive):
    say("\nNEGATIVE TYPES (How the negatives are generated)")

    if not negative_source:
        say("  no NA/RN column found - find how TChard encodes this")
        return

    negatives = work[~is_positive]
    counts = negatives[negative_source].value_counts(dropna=False)

    for value in counts.index:
        n = counts[value]
        row(value, f"{n:,}" + "  (" + f"{n / len(negatives):.1%}" + ")")

def report_lengths(work, peptide, b3):
    say("\nSEQUENCE LENGTHS")

    for column, limit in [(peptide, MAXLEN_PEP), (b3, MAXLEN_B3)]:
        lengths = work[column].astype(str).str.len()
        over = int((lengths > limit).sum())

        row(column,
            "min " + str(lengths.min())
            + "  p50 " + str(int(lengths.quantile(0.50)))
            + "  p99 " + str(int(lengths.quantile(0.99)))
            + "  max " + str(lengths.max())
            + "   limit " + str(limit)
            + "   over limit " + f"{over:,}")

def report_data_problems(df, work, peptide, b3):
    say("\nDATA PROBLEMS")

    for column in [peptide, b3]:
        bad_rows, characters = find_bad_residues(work[column])
        if bad_rows > 0:
            row(column + " bad residues",
                f"{bad_rows:,}" + " rows  (" + characters + ")")

    distinct_pairs = len(work[[peptide, b3]].drop_duplicates())
    labels_per_pair = work.groupby([peptide, b3])["_is_positive"].nunique()
    labels_per_b3 = work.groupby(b3)["_is_positive"].nunique()

    row("identical rows", f"{int(df.duplicated().sum()):,}")
    row("duplicate (peptide, B3) rows", f"{len(work) - distinct_pairs:,}")
    row("pairs with BOTH labels", f"{int((labels_per_pair > 1).sum()):,}")
    row("B3 seen with both labels",
        f"{int((labels_per_b3 > 1).sum()):,}"
        + "  of " + f"{len(labels_per_b3):,}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    for column in missing.index:
        row(column + " missing", f"{missing[column]:,}")

def report_gate(work, peptide, is_positive):
    say("\nGATE: epitopes by minimum positive rows")

    positive_counts = work[is_positive][peptide].value_counts()

    for threshold in [10, 20, 50, 100, 250]:
        row(">= " + str(threshold), f"{int((positive_counts >= threshold).sum()):,}")

    usable = int((positive_counts >= 50).sum())
    if usable >= 40:
        verdict = "PROCEED with per-epitope curves"
    else:
        verdict = "FALL BACK to coarse distance groups"

    share = work[peptide].value_counts().head(10).sum() / len(work)

def write_report(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "tchard_inspection.txt"
    path.write_text("\n".join(OUT) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", default="reports", type=Path)
    parser.add_argument("--peptide-col", default=None)
    parser.add_argument("--b3-col", default=None)
    parser.add_argument("--binder-col", default=None)
    parser.add_argument("--neg-source-col", default=None)
    parser.add_argument("--positive-value", default="1")
    args = parser.parse_args()

    df = load(args.input)
    say(f"{len(df):,}" + " rows x " + str(len(df.columns)) + " columns")
    say("columns: " + ", ".join(df.columns))

    peptide, b3, binder, negative_source = detect_columns(df, args)
    report_mapping(peptide, b3, binder, negative_source)

    if not peptide or not b3 or not binder:
        write_report(args)
        raise SystemExit("\nCould not detect a required column. "
                         "Set it with --peptide-col / --b3-col / --binder-col.")

    work = df.dropna(subset=[peptide, b3]).copy()
    work = work.reset_index(drop=True)
    is_positive = mark_positives(work[binder], args.positive_value)

    if is_positive.sum() == 0:
        write_report(args)
        raise SystemExit("\nNo rows match --positive-value "
                         + str(args.positive_value)
                         + ". Values present: "
                         + str(list(df[binder].unique())[:10]))

    work["_is_positive"] = is_positive

    report_summary(work, peptide, b3, is_positive)
    report_negative_types(work, negative_source, is_positive)
    report_lengths(work, peptide, b3)
    report_data_problems(df, work, peptide, b3)
    report_gate(work, peptide, is_positive)
    write_report(args)


if __name__ == "__main__":
    main()