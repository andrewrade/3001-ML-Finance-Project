import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import joblib
import preprocessor

def estimation(df_train, model_type=None, seed = 42):
    '''
     --- NaN
    '''
    if model_type == 'Logit':
        formula = 'Default ~ asset_turnover + leverage_ratio + interest_rate + roa + AR'
        model = sm.formula.logit(formula=formula, data=df_train)
        logit_model = model.fit()

        return logit_model
    
    elif model_type == 'Random_Forest':
        y_train = df_train['Default']
        X_train = df_train.drop('Default', axis=1)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        return clf
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Supported types are 'Logit' and 'Random_Forest'.")
    