from __future__ import annotations
import numpy as np
"""
Notes about Logistic Regression's encoder:

Each sequence is featurised into a 26-dimensional vector.
The first 20 dimensions are the amino-acid composition. This sums to 1. 
The 21st dimension is the sequence length.
The last 5 dimensions are mean physicochemical properties:
hydrophobicity, net charge, polarity, volume, aromaticity

Concat stacks the receptor and peptide vectors into 52 features. Tcr_only
keeps all 52 columns but zeroes the peptide half. This way, only the information
differs. Interaction appends the elementwise product of
the two halves (78 features) to capture pairwise effects.
"""
# List the amino acids 
AA = "ACDEFGHIKLMNPQRSTVWY"

# Create a dictionary that maps each amino acid to its index in the list
AA_INDEX = {}
for i, a in enumerate(AA):
    AA_INDEX[a] = i
N_AA = len(AA)
SEQ_DIMS = N_AA + 6
SETTINGS = ("tcr_only", "concat", "interaction")

# Utilized pre-existing Kyte-Doolittle hydropathy values
HYDROPATHY = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, 
              "F": 2.8, "G": -0.4, "H": -3.2,"I": 4.5, 
              "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, 
              "P": -1.6, "Q": -3.5,"R": -4.5, "S": -0.8, 
              "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3}

# net charge at physiological pH
# Got this from Lehninger Principles of Biochemistry, 7th edition
CHARGE = {"A": 0.0, "C": 0.0, "D": -1.0, "E": -1.0, 
          "F": 0.0, "G": 0.0, "H": 0.1, "I": 0.0,  
          "K": 1.0,  "L": 0.0, "M": 0.0, "N": 0.0, 
          "P": 0.0,  "Q": 0.0,  "R": 1.0, "S": 0.0, 
          "T": 0.0, "V": 0.0,  "W": 0.0,  "Y": 0.0}

# Utilized pre-existing Grantham polarity values
POLARITY = {"A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3,
             "F": 5.2, "G": 9.0, "H": 10.4, "I": 5.2, 
             "K": 11.3, "L": 4.9, "M": 5.7, "N": 11.6, 
             "P": 8.0, "Q": 10.5, "R": 10.5, "S": 9.2, 
             "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2}

# Utilized pre-existing Grantham volume values
VOLUME = {"A": 31.0, "C": 55.0, "D": 54.0, "E": 83.0, 
          "F": 132.0, "G": 3.0, "H": 96.0, "I": 111.0, 
          "K": 119.0, "L": 111.0, "M": 105.0, "N": 56.0,
          "P": 32.5, "Q": 85.0, "R": 124.0, "S": 32.0,
          "T": 61.0, "V": 84.0,"W": 170.0, "Y": 136.0}

# Got this from Lehninger Principles of Biochemistry, 7th edition
# Stores one if the amino acid is aromatic, zero otherwise
AROMATIC = {"A": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, 
            "F": 1.0, "G": 0.0, "H": 1.0, "I": 0.0, 
            "K": 0.0, "L": 0.0, "M": 0.0, "N": 0.0, 
            "P": 0.0, "Q": 0.0, "R": 0.0, "S": 0.0, 
            "T": 0.0, "V": 0.0, "W": 0.0, "Y": 1.0}

# Physicochemical scales stored in a tuple for easy iteration
_SCALES = (HYDROPATHY, CHARGE, POLARITY, VOLUME, AROMATIC)

# Convert a sequence into a 26-dimensional vector. 
def encode_sequence(seq: str) -> np.ndarray:
    # Check if the input is not NAN 
    if not isinstance(seq, str):
        seq = ""
    seq = seq.strip().upper()
    
    # Remove any non-amino acid characters
    residues = []
    for c in seq:
        if c in AA_INDEX:
            residues.append(c)

    # Create a vector of zeros 
    vec = np.zeros(SEQ_DIMS, dtype=np.float64)
    # If the sequence is empty, return the zero vector
    n = len(residues)
    if n == 0:
        return vec

    # Find the frequency of each amino acid in the sequence and store it in the vector
    for c in residues:
        vec[AA_INDEX[c]] += 1.0
    vec[:N_AA] /= n
    vec[N_AA] = float(n)

    # Compute the mean of each physicochemical scale and store it in the vector
    slot = N_AA + 1
    for scale in _SCALES:
        values = []
        for c in residues:
            values.append(scale[c])
        vec[slot] = float(np.mean(values))
        slot = slot + 1

    return vec

# Turn a list of sequences into an (n, 26) matrix, one row per sequence
def encode_many(seqs):
    if len(seqs) == 0:
        return np.zeros((0, SEQ_DIMS))

    rows = []
    for s in seqs:
        rows.append(encode_sequence(s))

    return np.vstack(rows)

# Build the design matrix based on the setting.  
def build_feature_matrix(tcr_seqs, pep_seqs, setting: str = "concat") -> np.ndarray:
    if setting not in SETTINGS:
        raise ValueError("setting must be one of " + str(SETTINGS) + ". Current value is " + str(setting))

    T = encode_many(list(tcr_seqs))
    P = encode_many(list(pep_seqs))

    # Check that the number of TCRs and peptides match. 
    # If not, raise an error.
    if T.shape[0] != P.shape[0]:
        raise ValueError(f"row mismatch: {T.shape[0]} TCRs vs {P.shape[0]} peptides")

    if setting == "tcr_only":
        return np.hstack([T, np.zeros_like(P)])
    if setting == "concat":
        return np.hstack([T, P])
    return np.hstack([T, P, T * P])

# Return the names of the features for each column in the design matrix
def feature_names(setting: str = "concat") -> list[str]:
    base = []
    for a in AA:
        base.append("aa_" + a)
    base.append("len")
    base.append("hydropathy")
    base.append("charge")
    base.append("polarity")
    base.append("volume")
    base.append("aromatic")

    # First append "tcr_" to the receptor features, and then "pep_" to the peptide features.
    names = []
    for b in base:
        names.append("tcr_" + b)
    for b in base:
        names.append("pep_" + b)

    # The interaction setting is the products of the two halves. 
    if setting == "interaction":
        for b in base:
            names.append("inter_" + b)

    return names