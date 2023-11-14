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

'''
df = pd.read_csv(r"/Users/chitvangoyal/Desktop/3001_project/3001-ML-Finance-Project/train.csv")
print(f"Number of records:{len(df):,}")
print("Preprocessing")

preproc_params = {
    "statement_offset" : 6,
    "ir_path": r"/Users/chitvangoyal/Desktop/3001_project/3001-ML-Finance-Project/ECB Data Portal_20231029154614.csv",
    "features": ['asset_turnover', 'leverage_ratio', 'roa','interest_rate', 'AR']
}


df_processed = preprocesser_func(df, preproc_params, new=True, interest_rates=True)
print(f"Number of records:{len(df_processed):,}")


df_processed.to_csv(r"/Users/chitvangoyal/Desktop/3001_project/3001-ML-Finance-Project/train_processed.csv")

print(df_processed.columns)

'''
preproc_params = {
    "statement_offset" : 6,
    "ir_path": "csv_files/ECB Data Portal_20231029154614.csv",
    "features": ['asset_turnover', 'leverage_ratio', 'roa','interest_rate', 'AR', 'ateco_industry', 'legal_struct','stmt_date', 'id']
}

df = pd.read_csv('csv_files/train.csv')
df.head()
start_index = df['stmt_date'].min()
model_type = 'Random_Forest'
model, test_stats_list, out_of_sample_stats_list = bootstrapped_walk_forward_harness(df, preprocessor_function = preprocessing_func, preproc_params=preproc_params, train_function = estimation, start_index=start_index, step_size=1, num_bootstrap_samples=10, model_type=model_type)

filename = 'rf_model.sav'
joblib.dump(model, filename)
