from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from encoding_lr import SETTINGS, build_feature_matrix
from encoding_lr import feature_names

"""
Notes about this file:
This is the code that runs Logistic-regression baselines 
"""

RANDOM_STATE = 0
max_iteration = 2000
input_solver = "lbfgs"
input_class_weight = "balanced"

# Creares a model that is a scaler followed by logistic regression.
def make_model(C=1.0):
    scaler = StandardScaler()

    logistic_regression = LogisticRegression(
        C=C,
        max_iter=max_iteration,
        solver=input_solver,
        class_weight=input_class_weight,
        random_state=RANDOM_STATE,
    )

    return Pipeline([("scale", scaler), ("lr", logistic_regression)])


# THis function takes in a training and evaluation dataframe, and returns a dataframe with predictions
def fit_and_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame, setting: str = "concat", C: float = 1.0) -> pd.DataFrame:
    # turn sequences into numbers
    # This variable stores the training input
    X_tr = build_feature_matrix(train_df["B3"], train_df["peptide"], setting)
    # This variable stores the training output
    y_tr = train_df["binder"].to_numpy()
    # This variable stores the test input
    X_ev = build_feature_matrix(eval_df["B3"], eval_df["peptide"], setting)

    # Train the model 
    model = make_model(C=C)
    model.fit(X_tr, y_tr)

    # Make predictions on the test set
    pred = model.predict_proba(X_ev)[:, 1]

    # Create a dataframe with the predictions
    predictions = eval_df[["row_id", "peptide", "B3", "binder"]].copy()
    predictions["pred"] = pred
    predictions["model"] = "lr"
    predictions["setting"] = setting

    return predictions


# This function returns a dataframe with the largest-magnitude coefficients
def coefficient_table(train_df: pd.DataFrame, setting: str, C: float, top: int) -> pd.DataFrame:
    from encoding_lr import feature_names
    
    # Train a model on the full training set
    X = build_feature_matrix(train_df["B3"], train_df["peptide"], setting)
    y = train_df["binder"].to_numpy()
    model = make_model(C=C)
    model.fit(X, y)

    # Pull the learned weights out of the logistic regression step
    logistic_regression = model.named_steps["lr"]
    weights = logistic_regression.coef_.ravel()

    # Pair each weight with the name of the feature it belongs to.
    table = pd.DataFrame({"feature": feature_names(setting), "coef": weights})

    # Sort by size. The largest (most impactful) is on the top
    table["size"] = table["coef"].abs()
    table = table.sort_values("size", ascending=False)
    table = table.drop(columns="size")