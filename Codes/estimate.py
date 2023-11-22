import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from preprocessor import remove_date_features
from utils import train_test_split_by_year

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
            # Take val as 20% of most recent year in data (prevent growing validation set size)
            train, val = train_test_split_by_year(df_train, date_column='stmt_date', test_frac=0.2)

            # Remove statement datetime column(s) from walk forward calls 
            train = remove_date_features(train)
            val = remove_date_features(val)

            y_train = train['Default']
            y_val = val['Default']
   
            X_train = train.drop('Default', axis=1)
            X_val = val.drop('Default', axis=1)

            clf = xgb.XGBClassifier(
                objective='binary:logistic',
                learning_rate=0.2,
                n_estimators=1000,
                colsample_bytree=0.8
            )
            
            clf.set_params(early_stopping_rounds=25, eval_metric="auc")
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
            return clf

        case _:
            raise ValueError(f"Invalid model_type: {model_type}. Supported types are 'Logit', 'Random_Forest' and 'XGboost'.")
        



        