import pandas as pd
import argparse
import joblib
from preprocessor import preprocessing_func
from prediction import predict_harness
import re 

def extract_encoded_feature_names(features):
    '''
    Extract the unique underlying features if there 
    are Onehot encoded features (ie XGBoost) 
    '''
    unencoded_features = set()
    regex = r'(_\d+)$'
    for feature in features:
        unencoded_feature = re.sub(regex, '', feature)
        unencoded_features.add(unencoded_feature)
    
    return list(unencoded_features)

def load_model(model_type):
    '''
    Loads model and preprocessing parameters 
    based on passed model type
    '''
    match model_type:
        
        case 'XGboost':
            model_file = 'models/xgb_model.sav'
            model = joblib.load(model_file)
            encoded_features = model.feature_names_in_
            print(encoded_features)
            features = extract_encoded_feature_names(encoded_features)
            one_hot_encode = True

        case 'Logit':
            model_file = 'models/basic_model.sav'
            model = joblib.load(model_file)
            features = model.params.index
            one_hot_encode = False
            
        case 'Random_Forest':
            model_file = 'models/rf_model.sav'
            model = joblib.load(model_file)
            features = model.feature_names_in_
            one_hot_encode = False
    
    return model, features, one_hot_encode

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, help="enter some quality limit",
                        nargs='?', default='csv_files/train.csv', const=0)
    parser.add_argument("--output_csv", type=str, help="enter some quality limit",
                        nargs='?', default='csv_files/pds_out.csv', const=0)
    args = parser.parse_args()

    input_file = args.input_csv
    output_file = args.output_csv

    '''
    Change model_type to run different pre-trained models. 
    Features and pre-processing parameters automatically 
    set based on selected model
    '''
    model_type='XGboost' # <<<<<< Change Model here 

    model, features, one_hot_encode = load_model(model_type)
    print(features)
    preproc_params = {
        "statement_offset" : 6,
        "ir_path": "csv_files/ECB Data Portal_20231029154614.csv",
        "features": features,
        "categorical_mapping_path":{
            'ateco_industry': 'csv_files/ateco_industry_mapping.csv',
            'legal_struct': 'csv_files/legal_struct_mapping.csv'
        }
    }

    test_raw = pd.read_csv(input_file).drop('def_date', axis=1)
    test_processed = preprocessing_func(test_raw, preproc_params, label=False, interest_rates=True, 
                                        one_hot_encode=one_hot_encode) # When selecting XGboost need to set `one_hot_encode` to True
    print(test_processed.columns)
    predictions = predict_harness(test_processed, model, model_type, plot_auc=False)

    pd.DataFrame({"PD":list(predictions)}).to_csv(output_file, index=False)


if __name__ == '__main__':
    main()