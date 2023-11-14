import pandas as pd
import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def default_check(row, date_range):
    '''
        Check if default occurs within the 12 month data range
    '''
    if pd.isnull(row['def_date']):
        return 0
    if row['def_date'] >= date_range[row['stmt_date']][0] and row['def_date'] < date_range[row['stmt_date']][1]:
        return 1
    return 0

def consolidate_ateco_codes(df):
    '''
    Consolidate ateco codes into industry groupings 
    (as defined in the ATECO 2007 code  reference)
    '''
    # Ateco code to industry mapping
    atc_cd = {1:"A",2:"A",3:"A",
          5:"B",6:"B",7:"B",8:"B",9:"B",
          10:"C",11:"C",12:"C",13:"C",14:"C",15:"C",16:"C",17:"C",18:"C",19:"C",20:"C",21:"C",22:"C",23:"C",24:"C",25:"C",26:"C",27:"C",28:"C",29:"C",30:"C",31:"C",32:"C",33:"C",
          35:"D",
          36:"E",37:"E",38:"E",39:"E",
          41:"F",42:"F",43:"F",
          45:"G",46:"G",47:"G",
          49:"H",50:"H",51:"H",52:"H",53:"H",
          55:"I",56:"I",
          58:"J",59:"J",60:"J",61:"J",62:"J",63:"J",
          64:"K",65:"K",66:"K",
          68:"L",
          69:"M",70:"M",71:"M",72:"M",73:"M",74:"M",75:"M",
          77:"N",78:"N",79:"N",80:"N",81:"N",82:"N",
          84:"O",
          85:"P",
          86:"Q",87:"Q",88:"Q",
          90:"R",91:"R",92:"R",93:"R",
          94:"S",95:"S",96:"S",
          97:"T",98:"T",
          99:"U",
          4: "V", # Adding additional industry mappings for the sector codes missing from the reference book
          34: "W",
          40: "X", 
          44: "Y",
          48: "Z",
          54: "AA",
          57: "AB",
          67: "AC",
          76: "AD",
          83: "AE",
          89: "AF"
          }
    
    df['ateco_industry'] = df['ateco_sector'].map(atc_cd)
    return df

def merge_interest_rates(df, preproc_params):
    '''
    Merge historical interest rate data into df.
    Interest rate is merged based on when statement data is available 
    (statement date + offset)    
    '''
    interest_rates = pd.read_csv(preproc_params['ir_path'])
    interest_rates['DATE'] = pd.to_datetime(interest_rates['DATE'], format="%m/%d/%Y")
    interest_rates.rename(columns={'ECB_Main_refinancing_operations': 'interest_rate'}, inplace=True)
        
    # Add offset to interest rate dates to merge with offset
    interest_rates['offset_date'] = interest_rates['DATE'] + pd.DateOffset(months=preproc_params['statement_offset'])
    # Merge as_of due to imprecision when adding date offset
    df = pd.merge_asof(df.sort_values("stmt_date"), interest_rates.sort_values("offset_date"), 
                       left_on='stmt_date', right_on='offset_date', direction='nearest')
    return df

def label_defaults(df, preproc_params):
    '''
    Assign binary classification labels
        1: Defaulted within 12 months from statement date + offset
        0: Did not default within 12 months
    '''
    df['def_date'] = pd.to_datetime(df['def_date'], format="%d/%m/%Y")
    
    date_range = dict()
    for date in df['stmt_date'].unique():
        date = pd.to_datetime(date)
        if pd.isnull(date):
            continue
        prediction_window_start = date + pd.DateOffset(months = preproc_params['statement_offset'])
        prediction_window_end = date + pd.DateOffset(years = 1, months = preproc_params['statement_offset'])
        date_range[date] = (prediction_window_start, prediction_window_end)
    
    df['Default'] = df.apply(lambda x: default_check(x, date_range), axis=1)
    return df

def financial_ratios(df, preproc_params):
    '''
    Compute the financial ratios passed into preproc_params
    Parameters:
        df: dataframe used to compute the financial ratios 
        preproc_params: dictionary, financial ratios to compute are passed into preproc_params['features]
    Returns:
        Dataframe including columns for each of the financial ratios passed 
    '''
    features = preproc_params['features']

    # OpEX required to calculate defensive interval 
    if 'operating_expenses' in features or 'defensive_interval' in features:
        df['operating_expenses'] = (df['rev_operating'] - df['prof_operations']) \
       .apply(lambda x: x if x > 0 else np.nan) # Operating expenses shouldn't be negative, set negative values to Nan

    # Current liabilities required to calculate Current and Quick ratios
    if 'current_liabilities' in features or 'current_ratio' in features or 'quick_ratio' in features:
        df['current_liabilities'] = (df['debt_bank_st'] + df['debt_fin_st'] + df['AP_st'] + df['debt_st'])
        df['current_liabilities'] = df['current_liabilities'].replace(0, 1) # Smoothing factor if current_liabilities = 0 

    #################################### Liquidity Ratios ##########################################
    
    if 'current_ratio' in features:
        # Current Ratio = Current Assets / Current Liabilities
        df['current_ratio'] = df['asst_current'] / (df['current_liabilities'])
    
    if 'quick_ratio' in features:
        # Quick Ratio = Immediate Short term Liquidity / Current Liabilities
        df['quick_ratio'] = df['cash_and_equiv'] / (df['current_liabilities'])

    if 'defensive_interval' in features:
        # Defensive interval (liquidity ratio) = liquid assets / daily cash burn
        df['defensive_interval'] = (df['cash_and_equiv'] + df['AR']) / ( df['operating_expenses'] / 365)

    ################################# Asset Management Ratios ######################################
    
    if 'asset_turnover' in features:
        # Asset turnover = rev_operations / asst_tot 
        df['asset_turnover'] = df['rev_operating'] / df['asst_tot']

    ###################################### Debt Ratios #############################################
    
    if 'debt_to_equity' in features:
        # Debt to equity = total debt / total equity
        df['debt_to_equity'] = (df['debt_st'] + df['debt_lt']) / (df['eqty_tot'])\
            .apply(lambda x: x if x != 0 else 1) # Smoothing factor if eqty_tot = 0
    
    # Debt to EBITDA = total debt / EDBITA
    if 'debt_to_ebitda' in features:
        df['debt_to_ebitda'] = (df['debt_st'] + df['debt_lt']) / df['ebitda']\
            .apply(lambda x: x if x != 0 else 1) # Smoothing factor if ebitda = 0
    
    if 'cfo_to_debt' in features:
        # CFO to debt ratio = Cash flow from operations / total debt
        df['cfo_to_debt'] = df['cf_operations'] / (df['debt_st'] + df['debt_lt'])\
            .apply(lambda x: x if x != 0 else 1) # Smoothing factor if debt = 0
    
    if 'cfo_to_op_earnings' in features:
        # CFO to operating earnings ratio = Cash flow from operations / operating_profit
        df['cfo_to_op_earnings'] = df['cf_operations'] / df['prof_operations']\
            .apply(lambda x: x if x != 0 else 1) # Smoothing factor if prof_operations = 0
    
    if 'leverage_ratio' in features:
        # Leverage ratio = total liabilities / total assets
        df['leverage_ratio'] = (df['asst_tot'] - df['eqty_tot']) / df['asst_tot']
    
    return df

def categorical_to_csv(df):
    '''
        Save mapping of categorical variables to encoded values to csv
    '''
    # Consolidate ateco sectors into industry groups
    df = consolidate_ateco_codes(df)
    df['ateco_industry'] = pd.Categorical(df['ateco_industry'])
    df['legal_struct'] = pd.Categorical(df['legal_struct'])

    ateco_industry_mapping = pd.DataFrame({
            'Original_Value': df['ateco_industry'],
            'Code': df['ateco_industry'].cat.codes
    })

    legal_struct_mapping = pd.DataFrame({
            'Original_Value': df['legal_struct'],
            'Code': df['legal_struct'].cat.codes
    })

    ateco_industry_mapping = ateco_industry_mapping.drop_duplicates()
    legal_struct_mapping = legal_struct_mapping.drop_duplicates()

    # Save the mappings to CSV
    legal_struct_mapping.to_csv('csv_files/legal_struct_mapping.csv', index=False)
    ateco_industry_mapping.to_csv('csv_files/ateco_industry_mapping.csv', index=False)
    
    return legal_struct_mapping, ateco_industry_mapping

def create_encoders_from_csv(mappings):
    '''
    Create one-hot encoder based off category codes
    stored in external csvs
    '''
    encoders = {}
    for feature, mapping_file in mappings.items():
        mapping_df = pd.read_csv(mapping_file)
        categories = mapping_df['Code'].unique()
        encoder = OneHotEncoder(categories=[categories], handle_unknown='ignore')
        encoders[feature] = encoder
    return encoders

def one_hot_encoder(df, preproc_params, categorical_features):
    '''
    One hot encode categorical variables for algorithms that don't directly 
    support categoricals (ie XGBoost)
    '''
    # Separate out the datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).copy()
    df_non_datetime = df.drop(datetime_cols.columns, axis=1)

    encoders = create_encoders_from_csv(preproc_params['categorical_mapping_path'])
    transformers = [(feature, encoders[feature] ,[feature]) for feature in categorical_features]
    column_transformer = ColumnTransformer(transformers, remainder='passthrough') # Transfrom only cat features, remainder pass through

    one_hot_encoded = column_transformer.fit_transform(df_non_datetime)
    
    if issparse(one_hot_encoded):
        one_hot_encoded = one_hot_encoded.toarray() # Column transformer outputs sparse matrix, convert to dense

    new_columns = []
    for feature in categorical_features:
        transform_name = feature
        feature_names = column_transformer.named_transformers_[transform_name].get_feature_names_out([feature])
        new_columns.extend(feature_names)
    
    remaining_columns = [col for col in df_non_datetime.columns if col not in categorical_features]
    all_columns = list(new_columns) + remaining_columns

    df_transformed = pd.DataFrame(one_hot_encoded, columns=all_columns, index=df_non_datetime.index) # Convert to df with all colum labels
    df_transformed = pd.concat([df_transformed, datetime_cols], axis=1)
    
    return df_transformed


def preprocessing_func(raw_df, preproc_params=None, label=True, interest_rates=True, one_hot_encode=False):
    '''
    Parameters:
        raw_df: Unpcrocessed dataframe
        preproc_params: Dictionary of parameters to use for preprocessor
            "statement_offset": (int) months offset to use for financial statement availability
            "ir_path": (str) path to the historical ECB interest rate csv
            "features": (list of strings) features to retain in the processed dataframe
            "categorical_mapping_path": (dictionary) this dictionary maps the categorical variables to 
                the csv path storing the translation between categorical variables and integer codes
        label: Boolean, True if add class labels
        interest_rates: Boolean, if True merge historical ECB interest rate data into df
    Returns:
        Processed dataframe for model training/inference
    '''
    # Set default preproc_params if none are provided 
    if preproc_params is None:
            preproc_params = {
        "statement_offset" : 6,
        "ir_path": "csv_files/ECB Data Portal_20231029154614.csv",
        "features": ['asset_turnover', 'leverage_ratio', 'roa','interest_rate', 'ateco_industry','AR'],
        "categorical_mapping_path":     {
                'ateco_industry': 'csv_files/ateco_industry_mapping.csv',
                'legal_struct': 'csv_files/legal_struct_mapping.csv'
            }
    }

    df = raw_df.copy()    
    df['stmt_date'] = pd.to_datetime(df['stmt_date'], format="%Y-%m-%d")

    # Compute any of the financial ratios passed into preproc_params['features]
    df = financial_ratios(df, preproc_params)
    
    if interest_rates: # Merge historical ECB interest rate data
        df = merge_interest_rates(df, preproc_params)

    if label: # Label defaulting firms
        feature_labels = df.columns # Save feature names
        df = label_defaults(df, preproc_params)
        all_labels = df.columns 
        default_label = [x for x in all_labels if x not in feature_labels] # Take difference between df labels to capture 'default' column label
        preproc_params['features'].append(default_label[0]) # Add default column label to features so that it isn't removed in next step

    # Drop df columns not being used (for sklearn classifiers)
    processed_df = df.drop(columns=[col for col in df.columns if col not in preproc_params['features']])

    categorical_features = [x for x in preproc_params['features'] if x in preproc_params['categorical_mapping_path'].keys()]

    # Check if any categorical variables are needed, do this after dropping columns since OH encoding introduces new feature names
    if len(categorical_features) > 0: 
        
        if 'ateco_industry' in preproc_params['features']:
            processed_df['ateco_industry'] = consolidate_ateco_codes(df)['ateco_industry'] # Create ateco_industry column by consolidating sector codes
        
        # Read categorical variable encodings from csv, or create the csvs if they don't already exist
        try: 
            legal_struct_mapping = pd.read_csv(preproc_params['categorical_mapping_path']['legal_struct'])
            ateco_industry_mapping = pd.read_csv(preproc_params['categorical_mapping_path']['ateco_industry'])
        except:
            legal_struct_mapping, ateco_industry_mapping = categorical_to_csv(df)
       
        if one_hot_encode: # One hot encode for XGboost
            processed_df = one_hot_encoder(processed_df, preproc_params, categorical_features)
        
        else: # Directly use categoricals for Logit, Random Forest
            
            if 'ateco_sector' in preproc_params['features']:
                processed_df['ateco_sector'] = pd.Categorical(df['ateco_sector'])
            
            # Map and encode 'legal_struct'
            if 'legal_struct' in preproc_params['features']:
                processed_df['legal_struct'] = processed_df['legal_struct'].\
                    map(dict(zip(legal_struct_mapping['Original_Value'], legal_struct_mapping['Code'])))
            
            # Map and encode 'ateco_sector'
            if 'ateco_industry' in preproc_params['features']:
                processed_df['ateco_industry'] = processed_df['ateco_industry'].\
                    map(dict(zip(ateco_industry_mapping['Original_Value'], ateco_industry_mapping['Code'])))
    
    return processed_df
    

def main():
    
    print("Loading training data")
    df = pd.read_csv("csv_files/train.csv")
    print(f"Number of records:{len(df):,}")
    print("Preprocessing")
    df_processed = preprocessing_func(df, label=True, interest_rates=True)
    print(f"Number of records:{len(df_processed):,}")
    df_processed.to_csv("csv_files/train_processed.csv")


if __name__ == "__main__":
    main()