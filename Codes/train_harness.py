from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocessor import preprocessing_func
from estimate import estimation
from prediction import predict_function
from walk_forward import bootstrapped_walk_forward_harness

preproc_params = {
        "statement_offset" : 6,
        "ir_path": "csv_files/ECB Data Portal_20231029154614.csv",
        "features": ['asset_turnover', 'leverage_ratio', 'roa','interest_rate', 'ateco_industry', 'stmt_date', 'id','AR'],
        "categorical_mapping_path": {
                'ateco_industry': 'csv_files/ateco_industry_mapping.csv',
                'legal_struct': 'csv_files/ateco_industry_mapping.csv'
            }
    }

df = pd.read_csv('csv_files/train.csv')
df.head()
start_index = df['stmt_date'].min()
model_type = 'XGboost'
model, test_stats_list, out_of_sample_stats_list = bootstrapped_walk_forward_harness(df, preprocessor_function = preprocessing_func, preproc_params=preproc_params, train_function = estimation, start_index=start_index, step_size=1, num_bootstrap_samples=10, model_type=model_type)

filename = 'xgb_model.sav'
joblib.dump(model, filename)
