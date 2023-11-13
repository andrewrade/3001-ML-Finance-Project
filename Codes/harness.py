import pandas as pd
import argparse
import joblib
from preprocessor import preprocessing_func, default_check, consolidate_ateco_codes, merge_interest_rates, label_defaults, financial_ratios, categorical_to_csv
from estimate import estimation
from prediction import predict_function
from walk_forward import bootstrapped_walk_forward_harness
from utils import predict_harness

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, help="enter some quality limit",
                    nargs='?', default='train.csv', const=0)
parser.add_argument("--output_csv", type=str, help="enter some quality limit",
                    nargs='?', default='train.csv', const=0)
args = parser.parse_args()

input_file = args.input_csv
output_file = args.output_csv

model_file = 'basic_model.sav'
model = joblib.load(model_file)
test = pd.read_csv(input_file).drop('def_date', axis=1)
model_type='Logit'

preproc_params = {
    "statement_offset" : 6,
    "ir_path": r"/Users/chitvangoyal/Desktop/3001_project/3001-ML-Finance-Project/ECB Data Portal_20231029154614.csv",
    "features": ['asset_turnover', 'leverage_ratio', 'roa','interest_rate', 'AR', 'stmt_date', 'id']
}

test = preprocessing_func(test, preproc_params, label=False, interest_rates=True)
predictions = predict_function(test_df = test, model= model, model_type = model_type)

pd.DataFrame({
            "PD":list(predictions)
            }).to_csv(output_file, index=False)
