import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Add parent directory to path to enable local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.evaluation.metrics import calculate_metrics

def evaluate_models(X_train, X_test, y_train, y_test, model_suffix=""):
    """
    Train and evaluate Linear Regression, Decision Tree, and Random Forest models.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_m = calculate_metrics(y_train, y_train_pred)
        test_m = calculate_metrics(y_test, y_test_pred)
        
        results[name] = {
            "Train": train_m,
            "Test": test_m
        }
        trained_models[name] = model
        
    return results, trained_models

def format_comparison_table(results):
    rows = []
    for model_name, sets in results.items():
        rows.append({
            "Model": model_name,
            "Train MAE": sets["Train"]["MAE"],
            "Test MAE": sets["Test"]["MAE"],
            "Train RMSE": sets["Train"]["RMSE"],
            "Test RMSE": sets["Test"]["RMSE"],
            "Train R2": sets["Train"]["R2"],
            "Test R2": sets["Test"]["R2"]
        })
    return pd.DataFrame(rows)

def plot_feature_importance(model, feature_names, title, save_path):
    """
    Plots the top feature importances of a model (Random Forest/Decision Tree) or Coefficients (Linear Regression).
    """
    plt.figure(figsize=(10, 6))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15] # Top 15 features
        
        sns.barplot(
            x=importances[indices],
            y=np.array(feature_names)[indices],
            palette="viridis"
        )
        plt.xlabel("Importance Score")
    else:
        # For Linear Regression, plot absolute coefficients
        coefs = np.abs(model.coef_)
        indices = np.argsort(coefs)[::-1][:15]
        
        sns.barplot(
            x=model.coef_[indices],
            y=np.array(feature_names)[indices],
            palette="coolwarm"
        )
        plt.xlabel("Coefficient Value")
        
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    processed_dir = 'data/processed'
    models_dir = 'models'
    assets_dir = 'src/app/assets'
    
    # Load targets
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()
    
    # ------------------ MODEL TRAINING ------------------
    print("\n" + "="*20 + " MODEL TRAINING (No G1/G2) " + "="*20)
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    
    results, models = evaluate_models(X_train, X_test, y_train, y_test)
    df_results = format_comparison_table(results)
    print("\nModel Performance Table:")
    print(df_results.to_string(index=False))
    
    # Save results table to CSV
    df_results.to_csv(os.path.join(processed_dir, 'results.csv'), index=False)
    
    # Select best model based on Test RMSE
    best_model_name = df_results.loc[df_results['Test RMSE'].idxmin()]['Model']
    print(f"\nBest Model: {best_model_name}")
    best_model = models[best_model_name]
    
    # Save Best Model
    joblib.dump(best_model, os.path.join(models_dir, 'model.pkl'))
    
    # Plot feature importances for Random Forest
    rf_model = models["Random Forest"]
    plot_feature_importance(
        rf_model, 
        X_train.columns, 
        "Random Forest - Top Feature Importances (No G1/G2)",
        os.path.join(assets_dir, 'feature_importance.png')
    )
    
    print("\nModel training, selection, and importance visualization complete!")

if __name__ == '__main__':
    main()
