import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

def predict_function(df, model=None, model_type='Logit'):
    '''
    returns probability
    not calculating AUC because hold out data will not have label column
    '''
    if model_type == 'Random_Forest':
        predictions = model.predict_proba(df)
        return(predictions[:, 1])
    elif model_type == 'Logit':
        predictions = model.predict(df)
        return(predictions)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Supported types are 'Logit' and 'Random_Forest'.")
    
def predict_harness(df, model, model_type, plot_auc=True):
    '''
    Prediction harness that produces probabilities of default (PD) using the passed model
    Parameters:
        df: dataframe to predict on. Dataframe expected to be pre-processed already
        model: The model used to predict PDs
        model_type: string, the type of model provided 'Logit' or 'Random_Forest'
        plot_auc: Boolean, whether to compute aucs in addition to predictions
    Returns:
        plot_auc == True:
            test['default']: (n x 1) series with class labels
            predictions: (n x 1) ndarray with PDs 
            auc_roc: auc_roc score based off predictions/class labels
        plot_auc == False:
            predictions: (n x 1) ndarray with PDs 
    '''
    predictions = predict_function(df, model=model, model_type=model_type)

    # Plot aucs when labels are known (during Walk forward validation)
    if plot_auc:
        new_prediction_df = pd.DataFrame({
            'Actual': df['Default'],
            'Predicted': predictions
        }).replace([np.inf, -np.inf], np.nan).dropna() # Drop Nans before computing auc
        
        actual_values = new_prediction_df['Actual']
        new_predictions =  new_prediction_df['Predicted']
        try:
            auc_roc = roc_auc_score(actual_values, new_predictions)
        except:
            auc_roc = np.nan
        return df['Default'], predictions, auc_roc

    return predictions