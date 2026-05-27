import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """Prepare features for modeling"""
    df_clean = df.copy()
    
    # Create target variables
    df_clean['ClaimStatus'] = (df_clean['TotalClaims'] > 0).astype(int)
    df_clean['LogClaimAmount'] = np.log1p(df_clean['TotalClaims'])
    
    # Encode categorical features
    cat_cols = ['Province', 'Gender', 'VehicleType']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le
    
    # Feature list
    feature_cols = ['Age', 'AnnualIncome', 'RiskScore', 'AnnualPremium', 
                    'Deductible', 'NCD', 'PastClaims'] + [c+'_encoded' for c in cat_cols]
    
    return df_clean, feature_cols, encoders
