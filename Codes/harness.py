import pandas as pd
import argparse
import joblib
from preprocessor import preprocessing_func

from prediction import predict_harness

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, help="enter some quality limit",
                    nargs='?', default='csv_files/train.csv', const=0)
parser.add_argument("--output_csv", type=str, help="enter some quality limit",
                    nargs='?', default='csv_files/pds_out.csv', const=0)
args = parser.parse_args()

input_file = args.input_csv
output_file = args.output_csv

model_type='Logit'

match model_type:
    
    case 'XGboost':
        model_file = 'xgb_model.sav'
        model = joblib.load(model_file)
        features = model.feature_names_in_
    
    case 'Logit':
        model_file = 'basic_model.sav'
        model = joblib.load(model_file)
        features = model.params.index
    
    case 'Random_Forest':
        model_file = 'rf_model.sav'
        model = joblib.load(model_file)
        features = model.feature_names_in_


preproc_params = {
    "statement_offset" : 6,
    "ir_path": "csv_files/ECB Data Portal_20231029154614.csv",
    "features": features,
    "categorical_mapping_path":{
        'ateco_industry': 'csv_files/ateco_industry_mapping.csv',
        'legal_struct': 'csv_files/legal_struct_mapping.csv'
    }
}

test = pd.read_csv(input_file).drop('def_date', axis=1)
test = preprocessing_func(test, preproc_params, label=False, interest_rates=True, 
                          one_hot_encode=False) # When selecting XGboost need to set `one_hot_encode` to True
predictions = predict_harness(test, model, model_type, plot_auc=False)

pd.DataFrame({
            "PD":list(predictions)
            }).to_csv(output_file, index=False)
