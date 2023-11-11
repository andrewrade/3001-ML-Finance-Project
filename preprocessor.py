import pandas as pd
import numpy as np

def default_check(row, date_range):
    if pd.isnull(row['def_date']):
        return 0
    if row['def_date'] >= date_range[row['stmt_date']][0] and row['def_date'] < date_range[row['stmt_date']][1]:
        return 1
    return 0

def preprocesser(raw_df, preproc_params, new=True, interest_rates=True):
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

    # Convert Legal Structure & ATECO Code to categorical fields 
    df['legal_struct'] = pd.Categorical(df['legal_struct'])
    df['legal_struct'] = df['legal_struct'].cat.codes
    df['ateco_sector'] = pd.Categorical(df['ateco_sector'])
    df['ateco_sector'] = df['ateco_sector'].cat.codes
    
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
    df['defensive_interval'] = (df['cash_and_equiv'] + df['AR'])\
        .apply(lambda x: x if x > 0 else np.nan) / ( df['operating_expenses'] / 365)

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
    df['cfo_to_op_earnings'] = df['cf_operations'] / df['prof_operations']
    
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

    return df
    

def main():
    
    print("Loading training data")
    df = pd.read_csv(r"C:\Users\Andrew Deur\Documents\NYU\DS-GA 3001 ML in Finance Discrete Choice\3001-ML-Finance-Project\train.csv")
    print(f"Number of records:{len(df):,}")
    print("Preprocessing")
    
    preproc_params = {
        "statement_offset" : 6,
        "ir_path": r"C:\Users\Andrew Deur\Documents\NYU\DS-GA 3001 ML in Finance Discrete Choice\3001-ML-Finance-Project\ECB Data Portal_20231029154614.csv",
    }

    df_processed = preprocesser(df, preproc_params, new=True, interest_rates=True)
    print(f"Number of records:{len(df_processed):,}")
    
    df_processed.to_csv(r"C:\Users\Andrew Deur\Documents\NYU\DS-GA 3001 ML in Finance Discrete Choice\3001-ML-Finance-Project\train_processed.csv")



if __name__ == "__main__":
    main()