from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def predict_function(test_df = None, model=None, model_type='Logit'):
    '''
    returns probability
    not calculating AUC because hold out data will not have label column
    '''

    if model_type == 'Random_Forest':
        predictions = model.predict_proba(test_df)
        return(predictions)
    elif model_type == 'Logit':
        predictions = model.predict(test_df)
        return(predictions)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Supported types are 'Logit' and 'Random_Forest'.")