import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import joblib

def estimation(df_train, formula = None, model_type=None, save_path = 'model.pkl', seed = 42):
    
    if model_type == 'Logit':
        model = sm.formula.logit(formula=formula, data=df_train)
        logit_model = model.fit()
        
        joblib.dump(logit_model, save_path)

        return logit_model
    
    elif model_type == 'Random_Forest':
        y_train = df_train['Default']
        X_train = df_train.drop('Default', axis=1)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        joblib.dump(clf, save_path)
        return clf
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Supported types are 'Logit' and 'Random_Forest'.")

