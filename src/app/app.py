import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to enable local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Register StudentPreprocessor class on the main execution module scopes to support pickling
from src.data.preprocess import StudentPreprocessor
for module_name in ['__main__', 'main']:
    if module_name in sys.modules:
        try:
            setattr(sys.modules[module_name], 'StudentPreprocessor', StudentPreprocessor)
        except Exception:
            pass

from src.models.predict import StudentPerformancePredictor

def load_css():
    """
    Inject custom CSS to establish premium styling, responsive layout, and glassmorphic cards.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* General resets and fonts */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main container padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Premium Card style */
        .glass-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        }
        
        /* Header Banner styling */
        .header-banner {
            background: linear-gradient(135deg, #1e293b 0%, #0f766e 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 25px;
            color: #f8fafc;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        .header-banner h1 {
            margin: 0;
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(90deg, #3b82f6, #14b8a6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-banner p {
            margin-top: 8px;
            margin-bottom: 0;
            color: #cbd5e1;
            font-size: 1.05rem;
            font-weight: 300;
        }
        
        /* Metric Box colors */
        .metric-risk {
            border-left: 5px solid #ef4444 !important;
            background: rgba(239, 68, 68, 0.08) !important;
        }
        .metric-pass {
            border-left: 5px solid #eab308 !important;
            background: rgba(234, 179, 8, 0.08) !important;
        }
        .metric-good {
            border-left: 5px solid #3b82f6 !important;
            background: rgba(59, 130, 246, 0.08) !important;
        }
        .metric-excellent {
            border-left: 5px solid #10b981 !important;
            background: rgba(16, 185, 129, 0.08) !important;
        }
        
        /* Customized forms */
        div[data-testid="stForm"] {
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background-color: rgba(30, 41, 59, 0.2) !important;
            border-radius: 12px !important;
        }
        
        /* Hover micro-animations on buttons */
        .stButton>button {
            background: linear-gradient(90deg, #0f766e, #14b8a6) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(20, 184, 166, 0.4) !important;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Student Performance Prediction Dashboard",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_css()
    
    # Header Banner
    st.markdown("""
        <div class="header-banner">
            <h1>🎓 Student Performance Analytics & Prediction</h1>
            <p>Predict final student grades (G3) using machine learning, analyzing demographics, social factors, and term grades.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize Predictor
    @st.cache_resource
    def get_predictor():
        return StudentPerformancePredictor(models_dir='models')
        
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Failed to load machine learning models. Please verify training scripts have run. Error: {e}")
        return

    # Main layout split
    col_input, col_results = st.columns([1.1, 0.9])
    
    with col_input:
        st.markdown('<div class="glass-card"><h3>📝 Student Attributes Form</h3>', unsafe_allow_html=True)
        
        # User input fields split into tabs for organization
        tab_edu, tab_dem, tab_soc = st.tabs(["📚 Academic & Support", "👤 Demographic", "🏡 Home & Habits"])
        
        with tab_edu:
            failures = st.slider("Past Class Failures", 0, 3, 0, help="Number of past class failures (0 to 3, or more)")
            studytime = st.selectbox(
                "Weekly Study Time",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "< 2 hours",
                    2: "2 to 5 hours",
                    3: "5 to 10 hours",
                    4: "> 10 hours"
                }[x]
            )
            schoolsup = st.selectbox("Extra Educational Support from School?", ["no", "yes"])
            famsup = st.selectbox("Family Educational Support?", ["no", "yes"])
            paid = st.selectbox("Extra Paid Classes (Tutoring)?", ["no", "yes"])
            
        with tab_dem:
            age = st.slider("Age", 15, 22, 17)
            sex = st.radio("Sex", ["Female", "Male"])
            school = st.radio("School", ["Gabriel Pereira (GP)", "Mousinho da Silveira (MS)"])
            address = st.radio("Address Type", ["Urban (U)", "Rural (R)"])
            famsize = st.radio("Family Size", ["Greater than 3 (GT3)", "Less than or equal to 3 (LE3)"])
            Pstatus = st.radio("Parent Cohabitation Status", ["Together (T)", "Apart (A)"])
            
        with tab_soc:
            internet = st.selectbox("Internet Access at Home?", ["yes", "no"])
            activities = st.selectbox("Extra-curricular Activities?", ["yes", "no"])
            romantic = st.selectbox("In a Romantic Relationship?", ["yes", "no"])
            nursery = st.selectbox("Attended Nursery School?", ["yes", "no"])
            higher = st.selectbox("Wants to Take Higher Education?", ["yes", "no"])
            
            st.markdown("---")
            st.subheader("Family & Environment Ratings (1-5)")
            famrel = st.slider("Quality of Family Relationships", 1, 5, 4, help="1: Very bad, 5: Excellent")
            freetime = st.slider("Free Time After School", 1, 5, 3, help="1: Very low, 5: Very high")
            goout = st.slider("Going Out with Friends", 1, 5, 3, help="1: Very low, 5: Very high")
            health = st.slider("Current Health Status", 1, 5, 5, help="1: Very bad, 5: Very good")
            
            st.markdown("---")
            st.subheader("Substance & Attendance")
            Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1, help="1: Very low, 5: Very high")
            Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1, help="1: Very low, 5: Very high")
            absences = st.number_input("School Absences", min_value=0, max_value=100, value=0)
            
            # Parental details
            Medu = st.slider("Mother's Education Level", 0, 4, 4, help="0: None, 4: Higher education")
            Fedu = st.slider("Father's Education Level", 0, 4, 3, help="0: None, 4: Higher education")
            Mjob = st.selectbox("Mother's Job", ["other", "teacher", "health", "services", "at_home"])
            Fjob = st.selectbox("Father's Job", ["other", "teacher", "health", "services", "at_home"])
            reason = st.selectbox("Reason to Choose School", ["course", "home", "reputation", "other"])
            guardian = st.selectbox("Student Guardian", ["mother", "father", "other"])
            traveltime = st.selectbox("Travel Time to School", [1, 2, 3, 4], format_func=lambda x: {
                1: "<15 min", 2: "15 to 30 min", 3: "30 min to 1h", 4: ">1h"
            }[x])

        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="glass-card"><h3>🔮 Prediction Analysis</h3>', unsafe_allow_html=True)
        
        # Compile input dict
        student_data = {
            'school': 'GP' if "GP" in school else 'MS',
            'sex': 'F' if sex == "Female" else 'M',
            'age': age,
            'address': 'U' if "Urban" in address else 'R',
            'famsize': 'GT3' if "Greater" in famsize else 'LE3',
            'Pstatus': 'T' if "Together" in Pstatus else 'A',
            'Medu': Medu,
            'Fedu': Fedu,
            'Mjob': Mjob,
            'Fjob': Fjob,
            'reason': reason,
            'guardian': guardian,
            'traveltime': traveltime,
            'studytime': studytime,
            'failures': failures,
            'schoolsup': schoolsup,
            'famsup': famsup,
            'paid': paid,
            'activities': activities,
            'nursery': nursery,
            'higher': higher,
            'internet': internet,
            'romantic': romantic,
            'famrel': famrel,
            'freetime': freetime,
            'goout': goout,
            'Dalc': Dalc,
            'Walc': Walc,
            'health': health,
            'absences': absences
        }
        
        # Run prediction
        predicted_grade = predictor.predict(student_data)
        
        # Determine status box type
        if predicted_grade >= 16.0:
            status_class = "metric-excellent"
            status_title = "Excellent Performance"
            status_desc = "The student is projected to excel. Model predicts a top-tier final grade. Keep encouraging active participation!"
            badge_color = "#10b981"
        elif predicted_grade >= 12.0:
            status_class = "metric-good"
            status_title = "Good Performance"
            status_desc = "The student is projected to perform well. General support is working. Monitor study habits to maintain progress."
            badge_color = "#3b82f6"
        elif predicted_grade >= 10.0:
            status_class = "metric-pass"
            status_title = "Pass / Average"
            status_desc = "The student is projected to achieve a passing grade. Some risk of falling below average. Recommend additional study hours."
            badge_color = "#eab308"
        else:
            status_class = "metric-risk"
            status_title = "At Risk of Failure"
            status_desc = "The student is projected to fail or score poorly. <b>Action recommended!</b> Target this student for academic tutoring and support interventions immediately."
            badge_color = "#ef4444"
            
        # Display Results Card
        st.markdown(f"""
            <div class="glass-card {status_class}" style="text-align: center; margin-top: 10px;">
                <h4 style="margin: 0; color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Final Grade (G3)</h4>
                <h1 style="margin: 10px 0; font-size: 4rem; font-weight: 700; color: {badge_color};">{predicted_grade:.2f} <span style="font-size: 1.5rem; color: #64748b;">/ 20</span></h1>
                <div style="background: rgba(255, 255, 255, 0.05); padding: 6px 12px; border-radius: 20px; display: inline-block; font-weight: 600; color: {badge_color}; margin-bottom: 12px;">
                    {status_title}
                </div>
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1; line-height: 1.4;">{status_desc}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Lower half: Model Insights & Feature Importance Tabs
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Model Insights")
        
        tab_fi, tab_eda = st.tabs(["🔑 Feature Weights", "📈 Dataset Distributions"])
        
        with tab_fi:
            st.write("These features had the most influence on the model's final predictions:")
            image_name = 'feature_importance.png'
            image_path = os.path.join('src/app/assets', image_name)
            
            if os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.info("Feature importance plot not found. Run model training first.")
                
        with tab_eda:
            st.write("General patterns found in the training dataset:")
            eda_option = st.selectbox(
                "Select Distribution Plot",
                ["Grade Distribution (G3)", "Failures vs G3", "Study Time vs G3", "Absences vs G3"]
            )
            
            eda_images = {
                "Grade Distribution (G3)": "grade_distribution.png",
                "Failures vs G3": "failures_vs_g3.png",
                "Study Time vs G3": "studytime_vs_g3.png",
                "Absences vs G3": "absences_vs_g3.png"
            }
            
            plot_path = os.path.join('src/app/assets', eda_images[eda_option])
            if os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)
            else:
                st.info("Distribution plot not found. Run EDA training script first.")
                
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
