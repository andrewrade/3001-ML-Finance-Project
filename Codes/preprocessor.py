import pandas as pd
import numpy as np
from utils import default_check


def preprocesser_func(raw_df, preproc_params, new=True, interest_rates=True):
    '''
    Parameters:
        raw_df: Raw data dataframe
        preproc_params: Dictionary of parameters to use for preprocessor
        new: Boolean, if True add class labels
        interest_rates: Boolean, if True merge historical ECB interest rate data into df
    
    Returns:
        Processed dataframe for model training/inference
    '''    
    df = raw_df.copy()
    df['stmt_date'] = pd.to_datetime(df['stmt_date'], format="%Y-%m-%d")
    
    # Merge historical ECB interest rate data
    if interest_rates:
        interest_rates = pd.read_csv(preproc_params['ir_path'])
        interest_rates['DATE'] = pd.to_datetime(interest_rates['DATE'], format="%m/%d/%Y")
        interest_rates.rename(columns={'ECB_Main_refinancing_operations': 'interest_rate'}, inplace=True)
        
        # Add 6 months to each statement date so joined interest has offset reflecting fin stmnt lag of ~6 months
        interest_rates['offset_date'] = interest_rates['DATE'] + pd.DateOffset(months=preproc_params['statement_offset'])
        df = pd.merge_asof(df.sort_values("stmt_date"), interest_rates.sort_values("offset_date"), 
                           left_on='stmt_date', right_on='offset_date', direction='nearest')

    ################## Ateco codes grouping #######################
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
          68:"l",
          69:"M",70:"M",71:"M",72:"M",73:"M",74:"M",75:"M",
          77:"N",78:"N",79:"N",80:"N",81:"N",82:"N",
          84:"O",
          85:"P",
          86:"Q",87:"Q",88:"Q",
          90:"R",91:"R",92:"R",93:"R",
          94:"S",95:"S",96:"S",
          97:"T",98:"T",
          99:"U"}
    df['ateco_sec'] = df['ateco_sector'].map(atc_cd)
    df['ateco_sec'] = pd.Categorical(df['ateco_sec'])
    df['legal_struct'] = pd.Categorical(df['legal_struct'])
    df['ateco_sector'] = pd.Categorical(df['ateco_sector'])

    if new:
        atc_cd_mapping = pd.DataFrame({
            'Original_Value': df['ateco_sec'],
            'Code': df['ateco_sec'].cat.codes
        })

        legal_struct_mapping = pd.DataFrame({
            'Original_Value': df['legal_struct'],
            'Code': df['legal_struct'].cat.codes
        })

        ateco_sector_mapping = pd.DataFrame({
            'Original_Value': df['ateco_sector'],
            'Code': df['ateco_sector'].cat.codes
        })

        atc_cd_mapping = atc_cd_mapping.drop_duplicates()
        legal_struct_mapping = legal_struct_mapping.drop_duplicates()
        ateco_sector_mapping = ateco_sector_mapping.drop_duplicates()

        # Save the mappings to CSV files
        atc_cd_mapping.to_csv('atc_cd_mapping.csv', index=False)
        legal_struct_mapping.to_csv('legal_struct_mapping.csv', index=False)
        ateco_sector_mapping.to_csv('ateco_sector_mapping.csv', index=False)
    
        ateco_sec_mapping = pd.read_csv('atc_cd_mapping.csv')
        legal_struct_mapping = pd.read_csv('legal_struct_mapping.csv')
        ateco_sector_mapping = pd.read_csv('ateco_sector_mapping.csv')


    # Map and encode 'ateco_sec'
    df['ateco_sec'] = df['ateco_sec'].map(dict(zip(ateco_sec_mapping['Original_Value'], ateco_sec_mapping['Code'])))

    # Map and encode 'legal_struct'
    df['legal_struct'] = df['legal_struct'].map(dict(zip(legal_struct_mapping['Original_Value'], legal_struct_mapping['Code'])))

    # Map and encode 'ateco_sector'
    df['ateco_sector'] = df['ateco_sector'].map(dict(zip(ateco_sector_mapping['Original_Value'], ateco_sector_mapping['Code'])))
    

    
    df['operating_expenses'] = (df['rev_operating'] - df['prof_operations'])\
        .apply(lambda x: x if x > 0 else np.nan) # Operating expenses shouldn't be negative, set negative values to Nan

    df['current_liabilities'] = (df['debt_bank_st'] + df['debt_fin_st'] + df['AP_st'] + df['debt_st'])
    df['current_liabilities'] = df['current_liabilities'].replace(0, 1) # Smoothing factor if current_liabilities = 0 

    #################################### Liquidity Ratios ##########################################
    
    # Current Ratio = Current Assets / Current Liabilities
    df['current_ratio'] = df['asst_current'] / (df['current_liabilities'])\
    
    # Quick Ratio = Immediate Short term Liquidity / Current Liabilities
    df['quick_ratio'] = df['cash_and_equiv'] / (df['current_liabilities'])
    
    # Defensive interval (liquidity ratio) = liquid assets / daily cash burn
    df['defensive_interval'] = (df['cash_and_equiv'] + df['AR']) / ( df['operating_expenses'] / 365)

    ################################# Asset Management Ratios ######################################
    
    # Asset turnover = rev_operations / asst_tot 
    df['asset_turnover'] = df['rev_operating'] / df['asst_tot']

    ###################################### Debt Ratios #############################################
    
    # Debt to equity = total debt / total equity
    df['debt_to_equity'] = (df['debt_st'] + df['debt_lt']) / (df['eqty_tot'])\
        .apply(lambda x: x if x != 0 else 1) # Smoothing factor if eqty_tot = 0
    
    # Debt to EBITDA = total debt / EDBITA
    df['debt_to_ebitda'] = (df['debt_st'] + df['debt_lt']) / df['ebitda']\
        .apply(lambda x: x if x != 0 else 1) # Smoothing factor if ebitda = 0
    
    # CFO to debt ratio = Cash flow from operations / total debt
    df['cfo_to_debt'] = df['cf_operations'] / (df['debt_st'] + df['debt_lt'])\
        .apply(lambda x: x if x != 0 else 1) # Smoothing factor if debt = 0
    
    # CFO to operating earnings ratio = Cash flow from operations / operating_profit
    df['cfo_to_op_earnings'] = df['cf_operations'] / df['prof_operations']\
        .apply(lambda x: x if x != 0 else 1) # Smoothing factor if prof_operations = 0
    
    # Leverage ratio = total liabilities / total assets
    df['leverage_ratio'] = (df['asst_tot'] - df['eqty_tot']) / df['asst_tot']

    # Assign class labels
    if new:
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

    # Drop df columns not being used (for sklearn classifiers)
    #processed_df = df.drop(columns=[col for col in df.columns if col not in preproc_params['features']])
    
    return df
    

def main():
    
    print("Loading training data")
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



if __name__ == "__main__":
    main()