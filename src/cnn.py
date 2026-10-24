
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset

from .encoding_cnn import (
    encode_sequence,
    MAXLEN_B3,
    MAXLEN_PEP,
)


# --------------------------------------------------
# 1. Dataset
# --------------------------------------------------

class TCREpitopeDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True).copy()

        required = {"B3", "peptide", "binder"}
        if not required.issubset(self.df.columns):
            raise ValueError(
                f"Missing columns: {required - set(self.df.columns)}"
            )

        self.tcrs = torch.stack([
            encode_sequence(seq, MAXLEN_B3)
            for seq in self.df["B3"]
        ])

        self.peptides = torch.stack([
            encode_sequence(seq, MAXLEN_PEP)
            for seq in self.df["peptide"]
        ])

        self.labels = torch.tensor(
            self.df["binder"].to_numpy(dtype=np.float32)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "tcr": self.tcrs[idx],
            "peptide": self.peptides[idx],
            "label": self.labels[idx],
        }


# --------------------------------------------------
# 2. Sequence CNN encoder
# --------------------------------------------------

class SequenceEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim=16,
        num_filters=64,
        kernel_size=3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=21,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        valid = x != 0

        x = self.embedding(x)

        x = x.transpose(1, 2)

        x = self.relu(self.conv(x))

        x = x.masked_fill(
            ~valid.unsqueeze(1),
            torch.finfo(x.dtype).min,
        )

        x = x.max(dim=2).values

        return x


# --------------------------------------------------
# 3. Dual-branch CNN
# --------------------------------------------------

class DualBranchCNN(nn.Module):
    def __init__(
        self,
        embedding_dim=16,
        num_filters=64,
        kernel_size=3,
        hidden_dim=64,
        dropout=0.3,
    ):
        super().__init__()

        self.tcr_encoder = SequenceEncoder(
            embedding_dim,
            num_filters,
            kernel_size,
        )

        self.peptide_encoder = SequenceEncoder(
            embedding_dim,
            num_filters,
            kernel_size,
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tcr, peptide):
        tcr_features = self.tcr_encoder(tcr)
        peptide_features = self.peptide_encoder(peptide)

        combined = torch.cat(
            [tcr_features, peptide_features],
            dim=1,
        )

        logits = self.classifier(combined)

        return logits.squeeze(-1)


# --------------------------------------------------
# 4. Training
# --------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=15,
    learning_rate=1e-3,
    device="cpu",
    patience=4,
    weight_decay=0.0
):
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(epochs):
        #Training
        model.train()
        total_loss = 0
        total_samples = 0

        for batch in train_loader:
            tcr = batch["tcr"].to(device)
            peptide = batch["peptide"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(tcr, peptide)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

        train_loss = total_loss / total_samples

        #Validation
        model.eval()
        val_total_loss = 0
        val_total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                tcr = batch["tcr"].to(device)
                peptide = batch["peptide"].to(device)
                labels = batch["label"].to(device)

                logits = model(tcr, peptide)
                loss = criterion(logits, labels)

                val_total_loss += loss.item() * len(labels)
                val_total_samples += len(labels)

        val_loss = val_total_loss / val_total_samples

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_weights)
    return model, history


# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------

@torch.no_grad()
def predict(model, data_loader, device="cpu"):
    model.eval()
    model.to(device)

    predictions = []

    for batch in data_loader:
        tcr = batch["tcr"].to(device)
        peptide = batch["peptide"].to(device)

        logits = model(tcr, peptide)
        probabilities = torch.sigmoid(logits)

        predictions.extend(
            probabilities.cpu().numpy().tolist()
        )

    return np.asarray(predictions)
