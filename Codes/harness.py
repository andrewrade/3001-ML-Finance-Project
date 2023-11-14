import pandas as pd
import argparse
import joblib
from preprocessor import preprocessing_func, default_check, consolidate_ateco_codes, merge_interest_rates, label_defaults, financial_ratios, categorical_to_csv
from estimate import estimation
from prediction import predict_function
from walk_forward import bootstrapped_walk_forward_harness
from prediction import predict_harness

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, help="enter some quality limit",
                    nargs='?', default='csv_files/train.csv', const=0)
parser.add_argument("--output_csv", type=str, help="enter some quality limit",
                    nargs='?', default='csv_files/pds_out.csv', const=0)
args = parser.parse_args()

input_file = args.input_csv
output_file = args.output_csv

model_file = 'rf_model.sav'
model = joblib.load(model_file)
test = pd.read_csv(input_file).drop('def_date', axis=1)
model_type='Random_Forest'

preproc_params = {
    "statement_offset" : 6,
    "ir_path": "csv_files/ECB Data Portal_20231029154614.csv",
    "features": ['asset_turnover', 'leverage_ratio', 'roa','interest_rate', 'AR', 'ateco_industry', 'legal_struct']
}

test = preprocessing_func(test, preproc_params, label=False, interest_rates=True)
predictions = predict_harness(test, model, model_type, plot_auc=False)

pd.DataFrame({
            "PD":list(predictions)
            }).to_csv(output_file, index=False)
