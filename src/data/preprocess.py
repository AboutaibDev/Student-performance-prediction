import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class StudentPreprocessor:
    def __init__(self):
        self.binary_mappings = {
            'school': {'GP': 1, 'MS': 0},
            'sex': {'F': 1, 'M': 0},
            'address': {'U': 1, 'R': 0},
            'famsize': {'GT3': 1, 'LE3': 0},
            'Pstatus': {'T': 1, 'A': 0},
            'schoolsup': {'yes': 1, 'no': 0},
            'famsup': {'yes': 1, 'no': 0},
            'paid': {'yes': 1, 'no': 0},
            'activities': {'yes': 1, 'no': 0},
            'nursery': {'yes': 1, 'no': 0},
            'higher': {'yes': 1, 'no': 0},
            'internet': {'yes': 1, 'no': 0},
            'romantic': {'yes': 1, 'no': 0}
        }
        self.nominal_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
        self.numeric_cols = [
            'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
            'failures', 'famrel', 'freetime', 'goout', 
            'Dalc', 'Walc', 'health', 'absences'
        ]
        
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.feature_names_ = None

    def fit(self, X):
        """
        Fit the OneHotEncoder and StandardScaler on the training data.
        """
        # Ensure we don't modify the original dataframe
        X = X.copy()
        
        # 1. Encode binary columns in-place to fit numeric scaling if needed
        for col, mapping in self.binary_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(0)
        
        # 2. Fit OneHotEncoder on nominal features
        self.ohe.fit(X[self.nominal_cols])
        
        # 3. Fit StandardScaler on numeric features
        self.scaler.fit(X[self.numeric_cols])
        
        # Determine feature names for transformed outputs
        ohe_features = self.ohe.get_feature_names_out(self.nominal_cols)
        binary_features = list(self.binary_mappings.keys())
        self.feature_names_ = self.numeric_cols + list(ohe_features) + binary_features
        return self

    def transform(self, X):
        """
        Transform the input data using the fitted mappings, encoder, and scaler.
        """
        X = X.copy()
        
        # 1. Map binary features
        for col, mapping in self.binary_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(0)
                
        # 2. Scale numeric features
        X_scaled = self.scaler.transform(X[self.numeric_cols])
        df_scaled = pd.DataFrame(X_scaled, columns=self.numeric_cols, index=X.index)
        
        # 3. One-hot encode nominal features
        X_ohe = self.ohe.transform(X[self.nominal_cols])
        ohe_features = self.ohe.get_feature_names_out(self.nominal_cols)
        df_ohe = pd.DataFrame(X_ohe, columns=ohe_features, index=X.index)
        
        # 4. Get binary features as dataframe
        df_binary = X[list(self.binary_mappings.keys())]
        
        # Concatenate everything
        X_transformed = pd.concat([df_scaled, df_ohe, df_binary], axis=1)
        
        # Reorder to match self.feature_names_
        X_transformed = X_transformed[self.feature_names_]
        return X_transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def main():
    # Paths
    raw_data_path = 'data/raw/student-mat.csv'
    processed_dir = 'data/processed'
    models_dir = 'models'
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading raw dataset...")
    df = pd.read_csv(raw_data_path, sep=';')
    
    # Simple validation checks
    print(f"Dataset shape: {df.shape}")
    assert 'G3' in df.columns, "Target column 'G3' is missing!"
    print(f"Missing values count: {df.isnull().sum().sum()}")
    
    # Train/Test Split (80% train, 20% test)
    # Target variable is G3. We drop G1 and G2 completely to prevent grade leakage.
    X = df.drop(columns=['G3', 'G1', 'G2'])
    y = df['G3']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set size: {X_train_raw.shape[0]} samples")
    print(f"Test set size: {X_test_raw.shape[0]} samples")
    
    # ------------------ PREPROCESSING ------------------
    print("\nProcessing student data...")
    preprocessor = StudentPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Save preprocessed sets
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))
    print("Preprocessed files and preprocessor saved successfully.")
    print("Preprocessing pipeline completed successfully!")

if __name__ == '__main__':
    main()
