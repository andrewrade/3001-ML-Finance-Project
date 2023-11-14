import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from pandas.api.types import is_datetime64_any_dtype

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
            # Builds statsmodel formula string by concatenating df column names 
            features = [x for x in df_train.columns if x != 'Default']
            formula = r"Default ~ " 
            for i, x in enumerate(features):
                formula += x.strip() 
                if i < len(features) - 1:
                    formula += " + "
            
            model = sm.formula.logit(formula=formula, data=df_train)
            logit_model = model.fit()

            return logit_model
            
        case 'Random_Forest':
            # Sklearn RF can't handle nans in input 
            df_train = df_train.dropna()
            y_train = df_train['Default']

            # Remove statement datetime column(s) from walk forward calls 
            feats_datetime_removed = [x for x in df_train.columns if not is_datetime64_any_dtype(df_train[x])]
            
            X_train = df_train[feats_datetime_removed]
            X_train = X_train.drop('Default', axis=1)
            
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            return clf
        
        case 'XGboost':
            pass
        
        
        case _:
            raise ValueError(f"Invalid model_type: {model_type}. Supported types are 'Logit' and 'Random_Forest'.")
        



        