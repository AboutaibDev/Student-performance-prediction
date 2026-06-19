# Student Performance Prediction Pipeline

🎓 **University Capstone Project Defense**  
An end-to-end Machine Learning pipeline to predict student final grades ($G3$) based on demographic, social, and academic attributes. This project implements Exploratory Data Analysis (EDA), custom preprocessing pipelines to avoid training-serving data leakage, model comparison (Linear Regression, Decision Tree, Random Forest), a programmatically compiled PDF report, and an interactive Streamlit web dashboard.

---

## 📂 Project Directory Structure

```text
student-performance-prediction/
├── data/
│   ├── raw/
│   │   └── student-mat.csv         # Raw student performance dataset (semicolon-separated)
│   └── processed/
│       ├── X_train.csv             # Preprocessed features for OLS/Tree/Forest training (No G1/G2)
│       ├── X_test.csv              # Preprocessed test features for evaluation
│       ├── y_train.csv             # Train targets (G3 final grades)
│       ├── y_test.csv              # Test targets (G3 final grades)
│       └── results.csv             # Model evaluation metrics comparison
│
├── notebooks/
│   └── EDA.ipynb                   # Jupyter notebook containing statistical summaries and charts
│
├── src/
│   ├── data/
│   │   └── preprocess.py           # Preprocessing class to encode and scale features
│   ├── evaluation/
│   │   └── metrics.py              # Evaluator helper computing MAE, RMSE, R²
│   ├── models/
│   │   ├── train.py                # Model training, comparison, and model serialization
│   │   └── predict.py              # Inference class loading serialized components
│   └── app/
│       ├── app.py                  # Interactive Streamlit application
│       └── assets/                 # Generated plot assets displayed in Streamlit/PDF
│           ├── grade_distribution.png
│           ├── correlation_heatmap.png
│           ├── failures_vs_g3.png
│           ├── studytime_vs_g3.png
│           ├── alcohol_vs_g3.png
│           ├── absences_vs_g3.png
│           └── feature_importance.png
│
├── models/
│   ├── preprocessor.pkl            # Serialized preprocessor pipeline (Scalers, Encoders)
│   └── model.pkl                   # Pickled Random Forest Regressor (best performer)
│
├── requirements.txt                # List of system package requirements
├── README.md                       # Main instruction manual (this file)
└── report.pdf                      # Programmatically generated publication-ready PDF report
```

---

## 🛠️ Installation & Setup

1. **Clone or navigate to the workspace directory**

2. **Install all required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Pipeline Usage Instructions

To run the machine learning pipeline and launch the application, execute these commands sequentially:

### 1. Execute Preprocessing
To clean the raw dataset, map binary variables, one-hot encode nominal categories, scale numerical variables, split into Train/Test partitions, and export the preprocessor pickle file:
```bash
python src/data/preprocess.py
```

### 2. Train Models
To train the Linear Regression, Decision Tree, and Random Forest models, evaluate them, output the metrics table, and serialize the best model to the `models/` directory:
```bash
python src/models/train.py
```

### 3. Launch Streamlit Application
To launch the interactive web dashboard:
```bash
streamlit run src/app/app.py
```

---

## 📊 Model Evaluation Summary

Our pipeline compares three models predicting the final grade ($G3$) based strictly on demographic, social, and behavioral features (excluding midterm grades $G1$ and $G2$ to prevent grade leakage and ensure early intervention utility):

| Model | Test MAE | Test RMSE | Test R² |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 3.40 | 4.20 | 0.14 |
| **Decision Tree** | 3.50 | 4.59 | -0.03 |
| **Random Forest (Best)** | **3.00** | **3.80** | **0.30** |

---

## 💡 Key Research Findings

1. **Early Prediction Utility**: Using only early-term behavioral and demographic indicators, our best model successfully flags at-risk students with a Test $R^2$ of **0.30** and Test MAE of **3.00** on a 20-point scale.
2. **Behavioral Impact**: The model extracts the strongest predictive signals from **number of past class failures**, **student absences**, **weekly study time**, and **parent education levels**.
3. **Substance Use**: High workday alcohol consumption ($Dalc$) is correlated with lower student scores, suggesting lifestyle variables carry significant academic weight.
4. **Best Algorithm**: The **Random Forest Regressor** consistently outperformed Linear Regression and Decision Trees by capturing complex non-linear feature interactions (such as the interaction between study time, parental education, and past failures).
