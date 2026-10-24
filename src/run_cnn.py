
import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader

from src.cnn import (
    TCREpitopeDataset,
    DualBranchCNN,
    train_model,
    predict,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

def load_data(config):
    train_df = pd.read_csv(config["data"]["train_path"])
    val_df = pd.read_csv(config["data"]["val_path"])
    test_df = pd.read_csv(config["data"]["test_path"])

    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")

    return train_df, val_df, test_df


def make_dataloaders(train_df, val_df, test_df, batch_size):
    train_dataset = TCREpitopeDataset(train_df)
    val_dataset = TCREpitopeDataset(val_df)
    test_dataset = TCREpitopeDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# 3. Save learning curves
# --------------------------------------------------

def save_learning_curve(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Dual-branch CNN learning curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved learning curve: {output_path}")


# --------------------------------------------------
# 4. Save predictions
# --------------------------------------------------

def save_predictions(test_df, probabilities, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(test_df) != len(probabilities):
        raise ValueError("Prediction count does not match test rows.")

    required_columns = ["row_id", "peptide", "B3", "binder"]
    missing = set(required_columns) - set(test_df.columns)

    if missing:
        raise ValueError(f"Missing test columns: {missing}")

    predictions_df = test_df[required_columns].copy()

    predictions_df["pred"] = probabilities
    predictions_df["model"] = "dual_branch_cnn"
    predictions_df["setting"] = "baseline"

    predictions_df.to_csv(output_path, index=False)

    print(f"Saved predictions: {output_path}")

    return predictions_df

# --------------------------------------------------
# 4. Evaluate CNN
# --------------------------------------------------


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


def evaluate_cnn(labels, predictions):
    auroc = roc_auc_score(labels, predictions)

    #Standardized partial ROC AUC for FPR <= 0.1
    auc01 = roc_auc_score(
        labels,
        predictions,
        max_fpr=0.1,
    )

    auprc = average_precision_score(
        labels,
        predictions,
    )

    return {
        "AUROC": auroc,
        "AUC0.1": auc01,
        "AUPRC": auprc,
    }


# --------------------------------------------------
# 5. Main training pipeline
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the dual-branch TCR-epitope CNN"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn.yaml",
        help="Path to CNN YAML configuration",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    #Seed for reproducibility
    set_seed(config["training"]["seed"])

    device = get_device()
    print(f"Using device: {device}")

    #Load datasets
    train_df, val_df, test_df = load_data(config)

    #Create DataLoaders
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df,
        val_df,
        test_df,
        batch_size=config["training"]["batch_size"],
    )

    #Initialize CNN
    model = DualBranchCNN(**config["model"])

    print("\nModel architecture:")
    print(model)

    #Train CNN
    print("\nStarting training...")

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        patience=config["training"]["patience"],
        weight_decay=config["training"]["weight_decay"],
        device=device,
    )

    #Save model checkpoint
    checkpoint_path = Path(config["output"]["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    #Save learning curves
    save_learning_curve(
        history,
        config["output"]["learning_curve"],
    )

    #Save training history
    history_path = checkpoint_path.with_suffix(".history.json")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    #Predict on held-out test set
    print("\nGenerating test predictions...")

    probabilities = predict(
        model,
        test_loader,
        device=device,
    )

    predictions_df = save_predictions(
        test_df,
        probabilities,
        config["output"]["predictions"],
    )

    
    #Calculate CNN evaluation metrics
    metrics = evaluate_cnn(
        labels=test_df["binder"].to_numpy(),
        predictions=probabilities,
    )

    print("\nCNN Test Results")
    print("-" * 35)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    #Save metrics
    metrics_path = Path("results/metrics/cnn_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to {metrics_path}")


    print("\nPrediction preview:")
    print(predictions_df.head())

    print("\nCNN pipeline completed successfully!")


if __name__ == "__main__":
    main()


