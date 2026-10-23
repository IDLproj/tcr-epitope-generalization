import argparse
from pathlib import Path
import pandas as pd

"""
Notes:

This file uses the NetTCR-2.2's dataset and splits it into three files for training, validation and testing.

This process utilizes the split provided by the authors of NetTCR-2.2. The purpose of this is to ensure thier 
models run with the data set they provided. This file will switch to use the TChard data set before the final deadline. 

The split occurs by using the "partition" column in the dataset, which contains values from 0 to 4. 
The script assigns these partitions to the appropriate splits: partition 4 is used for testing, partition 3 for validation, 
and partitions 0, 1, and 2 for training.

The authors grouped similar receptors together and kept each group inside a single partition. So no partition contains a 
receptor that closely resembles one in another partition.

To run it, use: python -m scripts.make_splits --input data/nettcr_2_2_limited_dataset.csv    
"""


# Got this from the authors github repo. It is the partition number they used for the test and validation set.
TEST_PARTITION = 4
VAL_PARTITION = 3

# The authors' dataset has 16 columns. Our project only neefs 4. 
# This list is written to make it clear which ones we use and what they mean.
# peptide - the epitope sequence
# B3- the CDR3-beta sequence
# binder - 1 = binds, 0 = does not bind
# partition - which of the authors groups this row belongs to
COLUMNS = ["peptide", "B3", "binder", "partition"]


# This function reads the CSV file. It only keeps important columns.
def load_dataset(path):
    df = pd.read_csv(path)

    missing = []
    for column in COLUMNS:
        if column not in df.columns:
            missing.append(column)

    if missing:
        raise SystemExit(f"These columns are missing from: {missing}\n")

    data = pd.DataFrame()
    data["row_id"] = df["index"] 
    data["peptide"] = df["peptide"].str.strip().str.upper()
    data["B3"] = df["B3"].str.strip().str.upper()
    data["binder"] = df["binder"].astype(int)
    data["partition"] = df["partition"].astype(int)
    return data


# Add a split column based on the partition number
def assign_splits(data):
    for i in data.index:
        if data.loc[i, "partition"] == VAL_PARTITION:
            data.loc[i, "split"] = "val"
        elif data.loc[i, "partition"] == TEST_PARTITION:
            data.loc[i, "split"] = "test"
        else:
            data.loc[i, "split"] = "train"
    return data


# This function prints the number of rows, binders and epitopes in each split. 
# It also prints the most and least common epitopes in the test set.
def describe(data):
    for split_name in ["train", "val", "test"]:
        rows = data[data["split"] == split_name]
        total_rows = len(rows)
        binders = int(rows["binder"].sum())
        percent_binders = 100 * binders / total_rows
        epitopes = rows["peptide"].nunique()

        print(split_name + ": " + str(total_rows) + " rows, " + str(binders) 
            + " binders (" + str(round(percent_binders, 1)) + "%), " + str(epitopes) + " epitopes")

    # Checks if there are any epitopes that dominate the test set.
    test_rows = data[data["split"] == "test"]
    counts = test_rows["peptide"].value_counts()
    most_common_epitope = counts.index[0]
    most_common_count = counts.iloc[0]
    least_common_epitope = counts.index[-1]
    least_common_count = counts.iloc[-1]
    print("Most rows: " + most_common_epitope + " with " + str(most_common_count))
    print("Fewest rows: " + least_common_epitope + " with " + str(least_common_count))

# Checks for any identical (peptide, B3) pairs in the train and test sets. 
# Note that this should be 0 sicne we should not see the same (peptide, B3) pair in both the train and test sets.
def check_for_leakage(data):
    train_rows = data[data["split"] == "train"]
    test_rows = data[data["split"] == "test"]
    train_pairs = set(zip(train_rows["peptide"], train_rows["B3"]))
    test_pairs = set(zip(test_rows["peptide"], test_rows["B3"]))
    repeated_pairs = train_pairs & test_pairs
    shared_receptors = set(train_rows["B3"]) & set(test_rows["B3"])
    shared_epitopes = set(train_rows["peptide"]) & set(test_rows["peptide"])

    print(f"Identical (peptide, B3) pairs in train and test: {len(repeated_pairs)}")
    print(f"CDR3-beta sequences in both: {len(shared_receptors)}")
    print(f"Epitopes in both: {len(shared_epitopes)}")

# Count repeated rows
# Since we model CDR3-beta only, two receptors differing only in their alpha
# chain become the same input. This function will report how many rows are 
# repeated in the dataset after we drop the alpha chain.
def report_duplicates(data):
    n_duplicates = int(data.duplicated(subset=["peptide", "B3", "binder"]).sum())
    if n_duplicates:
        print(f"Repeated (peptide, B3, binder) rows: {n_duplicates}")

 
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create the shared data splits")
    parser.add_argument(
        "--input",
        default="data/nettcr_2_2_limited_dataset.csv",
        help="path to the NetTCR-2.2 dataset CSV",
    )
    parser.add_argument(
        "--out-dir",
        default="splits",
        help="where to write train.csv, val.csv and test.csv",
    )
    args = parser.parse_args()

    # Load the dataset
    data = load_dataset(args.input)
    print(f"Read {len(data)} rows from {args.input}")

    # Assign each row to a split based on the partition number
    data = assign_splits(data)

    # Write the three splits to CSV files
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        rows = data[data["split"] == split_name].reset_index(drop=True)
        rows.to_csv(out_dir / f"{split_name}.csv", index=False)
    print("The train, validation, and test splits are written to the " + str(out_dir) + " directory")

    # Print some statistics about the splits and check for any issues
    # These are the expected outputs based on authors github: train - 20685, val- 6831, test - 6836
    describe(data)
    check_for_leakage(data)
    report_duplicates(data)


if __name__ == "__main__":
    main()