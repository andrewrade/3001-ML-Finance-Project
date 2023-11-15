import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from preprocessor import remove_date_features

def predict_function(df, model=None, model_type='Logit'):
    '''
    returns probability
    not calculating AUC because hold out data will not have label column
    '''
    match model_type:

        case 'Logit':
            predictions = model.predict(df)
            return(predictions)
        
        case 'Random_Forest':
            # Remove datetime columns and default column from walk forward calls
            feats_filtered = remove_date_features(df)
            feats_filtered = [x for x in df.columns if x != 'Default']
            
            # Skip rows where any featuers are nan, predict_proba method cannot handle nans
            nan_mask = df.isna().any(axis=1).to_numpy() # If any single feature is nan, set the records PD to nan 
            records_to_predict = df[~nan_mask]
            
            filtered_df = model.predict_proba(records_to_predict[feats_filtered]) 
            predictions = np.full((df.shape[0], model.n_classes_), np.nan) # Intialize predictions as nans
            stretched_mask = np.tile(nan_mask[:, np.newaxis], (1, model.n_classes_))
            predictions[~stretched_mask] = filtered_df.ravel() # Set non-nan indices to PDs computed in filtered_df

            return(predictions[:, 1])
        
        case 'XGboost':
            # Remove datetime columns and default column from walk forward calls
            feats_filtered = remove_date_features(df)
            feats_filtered = [x for x in df.columns if x != 'Default']
            
            X_train = df[feats_filtered]
            predictions = model.predict_proba(X_train)

            return(predictions[:, 1])
        
        case _:
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
    
    ###############################################################
    # TO DO: CALIBRATE PREDICTIONS USING METHODS IN OPTIMIZATION.PY
    ###############################################################

    return predictions