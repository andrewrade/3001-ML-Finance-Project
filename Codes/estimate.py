import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def remove_date_features(df):
    '''
        Remove datetime fields from walkforward analysis. 
        Datetime fields need to be removed before training tree based models
    '''
    keep = [x for x in df.columns if not is_datetime64_any_dtype(df[x])]
    return df[keep]

def estimation(df_train, model_type=None, seed = 42):
    '''
     Trains a model of the requested type
     Parameters:
        df_train: dataframe of the training data. Dataframe expected to be pre-processed already
        model_type: string, the type of model to train 'Logit', 'XGboost' or 'Random_Forest'
        seed: random seed
    Returns:
        Trained model of model_type
    '''
    match model_type:
        
        case 'Logit':
            features = [x for x in df_train.columns if x != 'Default'] # Remove label 
            formula = r"Default ~ " 
            for i, x in enumerate(features): # Builds statsmodel formula string by concatenating df column names 
                formula += x.strip() 
                if i < len(features) - 1:
                    formula += " + "
            
            model = sm.formula.logit(formula=formula, data=df_train)
            logit_model = model.fit()

            return logit_model
            
        case 'Random_Forest':
            df_train = df_train.dropna() # Sklearn RF can't handle nans in input 
            y_train = df_train['Default']

            # Remove statement datetime column(s) from walk forward calls 
            X_train = remove_date_features(df_train)
            X_train = X_train.drop('Default', axis=1)
            
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            return clf
        
        case 'XGboost':
            y_train = df_train['Default']
            X_train = df_train
            X_train = X_train.drop('Default', axis=1)

            # Remove statement datetime column(s) from walk forward calls 
            X_train = remove_date_features(X_train)

            print(X_train.columns)

            clf = xgb.XGBClassifier(
                objective='binary:logistic', 
                grow_policy='depthwise',
                seed=seed
                )
            
            clf.fit(X_train, y_train)
            return clf

        case _:
            raise ValueError(f"Invalid model_type: {model_type}. Supported types are 'Logit' and 'Random_Forest'.")
        



        