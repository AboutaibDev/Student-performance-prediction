import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE, and R2 score for given true and predicted values.
    
    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values
    
    Returns:
    dict: Dictionary containing MAE, RMSE, and R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def print_metrics_summary(model_name, y_train, y_train_pred, y_test, y_test_pred):
    """
    Prints a formatted summary of training and testing metrics for a model.
    """
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print(f"\n==================== {model_name} Evaluation ====================")
    print(f"{'Metric':<10} | {'Train Set':<12} | {'Test Set':<12}")
    print("-" * 45)
    for metric in ["MAE", "RMSE", "R2"]:
        print(f"{metric:<10} | {train_metrics[metric]:.4f}       | {test_metrics[metric]:.4f}")
    print("=" * 55)

def get_metrics_df(results_dict):
    """
    Creates a pandas DataFrame comparing metrics for different models.
    
    Parameters:
    results_dict (dict): Dictionary mapping model names to dictionaries of their metrics.
                         e.g., {"Linear Regression": {"Train": {...}, "Test": {...}}}
    
    Returns:
    pd.DataFrame: Table comparing models.
    """
    rows = []
    for model_name, sets in results_dict.items():
        for set_name, metrics in sets.items():
            rows.append({
                "Model": model_name,
                "Dataset": set_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"]
            })
    return pd.DataFrame(rows)
