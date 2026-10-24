
import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
#Assign indices to amino acids, starting from 1 (0 is reserved for padding)
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)} 

#Maximum lengths for TCR beta chain CDR3 and peptide sequences
MAXLEN_B3 = 23
MAXLEN_PEP = 12

#Encodes a sequence of amino acids into a tensor of indices, padding to the specified maximum length
def encode_sequence(sequence: str, max_length: int) -> torch.Tensor:
    sequence = sequence.strip().upper()
    encoded = [AA_TO_IDX[aa] for aa in sequence]
    padded = encoded + [0] * (max_length - len(encoded))
    return torch.tensor(padded, dtype=torch.long)
