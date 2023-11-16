import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from preprocessor import remove_date_features

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
            
            df_train = df_train.sort_values(by='stmt_date')
            
            cutoff = int(len(df_train) * 0.8)
            y = df_train['Default']
            y_train = y.iloc[:cutoff]
            y_val = y.iloc[cutoff:]
   
            # Remove statement datetime column(s) from walk forward calls 
            train = remove_date_features(df_train)
            train = train.drop('Default', axis=1)
            X_train = train[:cutoff]
            X_val = train[cutoff:]


            clf = xgb.XGBClassifier(
                objective='binary:logistic',
                learning_rate=0.2,
                n_estimators=1000,
                colsample_bytree=0.8
            )
            
            clf.set_params(early_stopping_rounds=10, eval_metric="auc")
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
            return clf

        case _:
            raise ValueError(f"Invalid model_type: {model_type}. Supported types are 'Logit', 'Random_Forest' and 'XGboost'.")
        



        