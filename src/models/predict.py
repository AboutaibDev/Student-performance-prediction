import os
import numpy as np
import pandas as pd
import joblib

# Add parent directory to path to enable local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.preprocess import StudentPreprocessor

# Ensure the preprocessor class is registered in the appropriate execution modules.
# This prevents deserialization failures when loading pickles that were generated
# under a different __main__ execution environment (like Streamlit which executes under the name 'main').
for module_name in ['__main__', 'main']:
    if module_name in sys.modules:
        try:
            setattr(sys.modules[module_name], 'StudentPreprocessor', StudentPreprocessor)
        except Exception:
            pass

class StudentPerformancePredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.preprocessor = None
        self.model = None
        self._load_components()

    def _load_components(self):
        """
        Load the fitted preprocessor and trained model.
        """
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        model_path = os.path.join(self.models_dir, 'model.pkl')

        if os.path.exists(preprocessor_path) and os.path.exists(model_path):
            self.preprocessor = joblib.load(preprocessor_path)
            self.model = joblib.load(model_path)

    def predict(self, student_data):
        """
        Predict the final grade (G3) for a student.
        
        Parameters:
        student_data (dict): Dictionary containing the student features.
        
        Returns:
        float: Predicted grade (bounded between 0 and 20)
        """
        # Load components dynamically if not already loaded
        if self.model is None or self.preprocessor is None:
            self._load_components()

        # Convert single dictionary to DataFrame
        df_input = pd.DataFrame([student_data])
        
        # Drop G1 and G2 if they happen to be in the input dictionary
        if 'G1' in df_input.columns:
            df_input = df_input.drop(columns=['G1'])
        if 'G2' in df_input.columns:
            df_input = df_input.drop(columns=['G2'])
            
        if self.model is None or self.preprocessor is None:
            raise FileNotFoundError("Model or preprocessor components have not been trained or saved yet!")
        
        # Preprocess and predict
        df_processed = self.preprocessor.transform(df_input)
        prediction = self.model.predict(df_processed)[0]
            
        # Constrain predictions to the valid grade range [0, 20]
        prediction = np.clip(prediction, 0.0, 20.0) if hasattr(prediction, 'item') else max(0.0, min(20.0, prediction))
        return float(prediction)

# Test function
if __name__ == '__main__':
    # Define a sample student
    sample_student = {
        'school': 'GP', 'sex': 'F', 'age': 16, 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T',
        'Medu': 4, 'Fedu': 3, 'Mjob': 'services', 'Fjob': 'other', 'reason': 'reputation',
        'guardian': 'mother', 'traveltime': 1, 'studytime': 2, 'failures': 0, 'schoolsup': 'no',
        'famsup': 'yes', 'paid': 'no', 'activities': 'yes', 'nursery': 'yes', 'higher': 'yes',
        'internet': 'yes', 'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 2,
        'Dalc': 1, 'Walc': 1, 'health': 5, 'absences': 4
    }
    
    try:
        predictor = StudentPerformancePredictor()
        pred = predictor.predict(sample_student)
        print(f"Sample Student Prediction: {pred:.2f}")
    except Exception as e:
        print(f"Prediction test failed (expected if models not trained yet): {e}")
