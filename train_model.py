import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pickle

# --- Data Preprocessing Function ---
def preprocess_data(df, drop_col, target=None, recode_map=None):
    """
    Drops the specified column and creates a binary target column if provided.
    """
    df = df.drop(columns=[drop_col], errors='ignore')
    if target and recode_map:
        df = df.dropna(subset=[target]).copy()
        df[target + '_binary'] = df[target].map(recode_map)
    return df

# --- Hyperparameter Search Space for Bayesian Optimization ---
param_space = {
    'max_depth': Integer(1, 500),                   # Integer between 1 and 500
    'learning_rate': Real(0.01, 1.3, prior='uniform'),  # Real values between 0.01 and 1.3
    'n_estimators': Integer(50, 1000)               # Integer between 50 and 1000
}

# --- Function to Train and Save Model ---
def train_and_save_model(df, drop_col, target, recode_map, predictors, model_filename):
    # Preprocess the dataset
    df_processed = preprocess_data(df, drop_col, target, recode_map)
    target_binary = target + '_binary'
    
    # Drop rows with missing predictor or target values
    data = df_processed.dropna(subset=predictors + [target_binary]).copy()
    X = data[predictors]
    y = data[target_binary]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize the model (XGBoost naturally handles missing values)
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    
    # Hyperparameter Tuning via Bayesian Optimization
    search = BayesSearchCV(model, param_space, n_iter=20, cv=5, scoring='roc_auc', 
                           random_state=42, n_jobs=-1)
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Evaluate Model Performance
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc_val = roc_auc_score(y_test, y_pred_proba)
    print(f"Training complete for {model_filename}")
    print(f"Best Hyperparameters: {search.best_params_}")
    print(f"AUC on Test Set: {auc_val:.4f}")
    
    # Save the Trained Model
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Trained model saved as '{model_filename}'\n")
    
    return best_model

# --- Train for Ewes ---
# Update file path as needed
ewes_df = pd.read_csv("ewes.csv")
ewes_predictors = ['age', 'av70dwt', 'avbirthwt', 'birth', 'born', 
                   'cumweaned', 'fertility', 'lambings', 'tot70dwt', 'years of breeding use']
ewes_target = 'group'
ewes_recode_map = {'minus': 0, 'plus': 1}

print("Training model for Ewes...")
ewes_model = train_and_save_model(ewes_df, 'Unnamed: 0', ewes_target, ewes_recode_map, 
                                  ewes_predictors, "xgb_model_ewes.pkl")

# --- Train for Lambs ---
# Update file path as needed
lambs_df = pd.read_csv("lambs.csv")
lambs_predictors = ['eweage', 'lambbirthwt', 'lambday70wt', 'lambgrowthrate',
                    'livelambbirthtype', 'mort', 'mortafter2', 'mortbefore70']
lambs_target = 'eweseroconverted'
lambs_recode_map = {'never': 0, 'before': 1, 'during': 1}

print("Training model for Lambs...")
lambs_model = train_and_save_model(lambs_df, 'Unnamed: 0', lambs_target, lambs_recode_map, 
                                   lambs_predictors, "xgb_model_lambs.pkl")
