import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier  # Ensure xgboost is installed

# ---------------------------
# Helper: Load Pre-Trained Model
# ---------------------------
@st.cache_data
def load_model(model_file):
    subprocess.run(["python", "train_model.py"], check=True)
    st.success("Model trained successfully and saved as 'xgb_model_ewes.pkl'")
    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file {model_file} not found. Please train and save the model first.")
        return None

# ---------------------------
# Predictor Definitions
# ---------------------------
ewes_predictors = ['age', 'av70dwt', 'avbirthwt', 'birth', 'born', 
                   'cumweaned', 'fertility', 'lambings', 'tot70dwt', 'years of breeding use']

lambs_predictors = ['eweage', 'lambbirthwt', 'lambday70wt', 'lambgrowthrate', 
                    'livelambbirthtype', 'mort', 'mortafter2', 'mortbefore70']

# ---------------------------
# Streamlit Interface
# ---------------------------
st.title("Seroconversion Prediction Tool :sheep:")
st.write("Enter the values you have. If a value is not available, leave the field blank.")

# Select dataset type
dataset_type = st.radio("Select Dataset Type", ("Ewes", "Lambs"))

# Load the appropriate model based on selection
if dataset_type == "Ewes":
    model_file = "xgb_model_ewes.pkl"
    predictors = ewes_predictors
    st.header("Enter Ewe Data")
else:
    model_file = "xgb_model_lambs.pkl"
    predictors = lambs_predictors
    st.header("Enter Lamb Data")

model = load_model(model_file)

# Create an input form for the predictors.
input_data = {}
for feature in predictors:
    value = st.text_input(f"{feature} (leave blank if not available)", "")
    # If the field is left blank, assign a missing value (np.nan)
    if value == "":
        input_data[feature] = np.nan
    else:
        try:
            # Convert input to float (adjust if your predictors require different types)
            input_data[feature] = float(value)
        except ValueError:
            st.error(f"Invalid value for {feature}. Please enter a numeric value.")
            st.stop()  # Stop execution if input is invalid

# When the user clicks the Predict button, make a prediction.
if st.button("Predict Seroconversion :racehorse:"):
    if model is None:
        st.error("Model could not be loaded.")
    else:
        # Create a DataFrame with one row from the provided inputs.
        input_df = pd.DataFrame([input_data])
        
        # Predict probability using the loaded model.
        try:
            prob = model.predict_proba(input_df)[:, 1][0]
            st.subheader(f"Predicted Probability of Seroconversion: {prob:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# ---------------------------
# Instructions for Deployment
# ---------------------------
st.markdown("""
### Deployment Instructions

1. **Save the code** in a file named `app.py`.

2. **Model files:** Ensure you have trained models saved as `xgb_model_ewes.pkl` and `xgb_model_lambs.pkl` in the same directory as `app.py`.  
   (If you wish to use a different model or adjust the logic, modify the code accordingly.)

3. **Install required packages:**  
   Run:  
   ```bash
   pip install streamlit xgboost pandas numpy""")
