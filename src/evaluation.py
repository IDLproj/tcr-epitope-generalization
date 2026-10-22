"""
Evaluation for all three models. 

Input: predictions (csv)
Output: results/summary.csv (per model and setting) & results/per_epitope.csv


"""

def validate(df):
    """
    Check the predictions CSV has the right columns and valid values.
    """
    pass

def evaluate_metrics(y_true, y_pred):
    """
    Takes (y_true, y_pred) and return a dict with auc/ auc01/ aupr.
    """
    metrics = {}

    return metrics # dict

def per_epitope(df):
    """
    Metrics for each peptide. return a DataFrame and one row per epitope.
    """
    pass


def evaluate(df):
    """
    Metrics for each (model, setting). 
    Returns (summary, per_epitope) DataFrames.
    """
    pass

def main():
    pass


if __name__ == "__main__":
    main()