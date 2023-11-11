from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def progress_bar(k, n, incr_txt="Step", bar_len = 10):
    bar_char  = u'\u25A5'
    line_char = u'\u21E2' # u'\u2192'  u'\u23AF' u'\u25AD'
    pct      = k/n
    n_str    = "{:,.0f}".format(n)
    k_str    = "{:,.0f}".format(k)
    pct_str  = "{:,.0f}%".format(k/n * 100)
    if k == n-1:
        n_bars = bar_len
        n_spaces = 0
        text_txt = "Completed " + n_str + " " + incr_txt + "s."
    else:
        n_bars   = int(np.floor(pct * bar_len))
        n_spaces = bar_len - n_bars
        text_txt = " " + pct_str +  " (" + incr_txt + " " + k_str + " of  " + n_str + ")."
    bar_txt  = "[" + "".ljust(n_bars,bar_char) + "".rjust(n_spaces,line_char) + "]  "
    clear_output()
    display(bar_txt + text_txt)

def default_check(row, date_range):
    if pd.isnull(row['def_date']):
        return 0
    if row['def_date'] >= date_range[row['stmt_date']][0] and row['def_date'] < date_range[row['stmt_date']][1]:
        return 1
    return 0

def stratified_split(df, label):
    grouped_df = df.groupby('id').agg({'id': 'count', label: 'max'})
    grouped_df = grouped_df.rename(columns={'id': 'count', label: 'default_max'})
    grouped_df = grouped_df.reset_index()

    in_sample_companies, out_of_sample_companies = train_test_split(df, test_size=0.3, random_state=42)
    in_sample_companies = in_sample_companies['id'].values
    out_of_sample_companies = out_of_sample_companies['id'].values

    return df[df.id.isin(in_sample_companies)], df[df.id.isin(out_of_sample_companies)]

def predict_harness(test, model, predictor_function):
    predictions = predictor_function(test, model)

    new_prediction_df = pd.DataFrame({
        'Actual': test['Default'],
        'Predicted': predictions
    }).replace([np.inf, -np.inf], np.nan).dropna()
    actual_values = new_prediction_df['Actual']
    new_predictions =  new_prediction_df['Predicted']

    try:
        auc_roc = roc_auc_score(actual_values, new_predictions)
    except:
        auc_roc = np.nan
        
    return test['Default'], predictions, auc_roc

def plot_auc_rocs(actual_values, predictions, out_of_sample_truth=None, out_of_sample_predictions=None):
    new_prediction_df = pd.DataFrame({
        'Actual Values': actual_values,
        'Predictions': predictions
    }).replace([np.inf, -np.inf], np.nan).dropna()

    actual_values =  new_prediction_df['Actual Values']
    predicted_probabilities =  new_prediction_df['Predictions']

    auc_roc = roc_auc_score(actual_values, predicted_probabilities)
    fpr, tpr, thresholds = roc_curve(actual_values, predicted_probabilities)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Test ROC curve (area = {:.2f})'.format(auc_roc))
    if out_of_sample_predictions:
      new_prediction_df = pd.DataFrame({
          'Actual Values': out_of_sample_truth,
          'Predictions': out_of_sample_predictions
      }).replace([np.inf, -np.inf], np.nan).dropna()
      actual_values =  new_prediction_df['Actual Values']
      predicted_probabilities =  new_prediction_df['Predictions']
      auc_roc = roc_auc_score(actual_values, predicted_probabilities)
      fpr, tpr, thresholds = roc_curve(actual_values, predicted_probabilities)
      plt.plot(fpr, tpr, color='green', lw=2, label='Val ROC curve (area = {:.2f})'.format(auc_roc))
    plt.plot([0, 1], [0, 1], lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()

def walk_forward_harness(df, preprocessor_function, train_function, predictor_function, start_index, step_size=1):
    df, preproc_params = preprocessor_function(df)
    
    label='Default'
    step_col='stmt_date'
    start_index = pd.to_datetime(start_index)
    df[step_col] = pd.to_datetime(df[step_col])

    steps = [start_index]
    step = start_index+pd.DateOffset(years= 1)
    while step<df[step_col].max():
        steps.append(step)
        step = step+pd.DateOffset(years= 1)
    n = len(steps)

    test_stats_list = []
    test_truth = []
    test_predictions = []
    out_of_sample_stats_list = []
    out_of_sample_truth = []
    out_of_sample_predictions = []

    train_df, validation_df = stratified_split(df, label)

    for k, step in enumerate(steps):
        progress_bar(k, n, incr_txt="Step", bar_len = 10)
        
        train = train_df[train_df[step_col] <= step]

        test = train_df[train_df[step_col] >= step]
        test = test[test[step_col] < step+pd.DateOffset(years= 1)]

        val = validation_df
        if k>0:
            val = validation_df[validation_df[step_col] >= step]
        if k<n-1:
            val = val[val[step_col] < step+pd.DateOffset(years= 1)]
          
        # print(train.shape, test.shape, val.shape)

        model = train_function(train)

        if test.shape[0]>0:
          actual_values, predictions, stats = predict_harness(test, model, predictor_function)
          test_stats_list.append(stats)
          test_truth += list(actual_values)
          test_predictions += list(predictions)
        else:
          print(k,n)

        if val.shape[0]>0:
          actual_values, predictions, stats = predict_harness(val, model, predictor_function)
          out_of_sample_stats_list.append(stats)
          out_of_sample_truth += list(actual_values)
          out_of_sample_predictions += list(predictions)
        else:
          print(k,n)

    plot_auc_rocs(test_truth, test_predictions, out_of_sample_truth, out_of_sample_predictions)
    plt.show()

    model = train_function(df)
    return model, test_stats_list, out_of_sample_stats_list, preproc_params

def trial_preprocessor(df, preproc_params={'offset': 5}, new=True):
    df['stmt_date'] = pd.to_datetime(df['stmt_date'], format='%Y-%m-%d')
    df['def_date'] = pd.to_datetime(df['def_date'], format='%d/%m/%Y')
    date_range = dict()
    for date in df['stmt_date'].unique():
        date = pd.to_datetime(date)
        if pd.isnull(date):
          continue
        prediction_window_start = date + pd.DateOffset(months = preproc_params['offset'])
        prediction_window_end = date + pd.DateOffset(years = 1, months = preproc_params['offset'])
        date_range[date] = (prediction_window_start, prediction_window_end)
    df['Default'] = df.apply(lambda x: default_check(x, date_range), axis=1)

    df['A'] = df['wc_net']/df['asst_tot']
    df['B'] = df['profit']/df['asst_tot']
    df['C'] = df['ebitda']/df['asst_tot']
    denominator = df['asst_tot'] - df['eqty_tot']
    df['D'] = df['eqty_tot'] / denominator.where(denominator != 0, 0.01)
    df['E'] = df['rev_operating']/df['asst_tot']

    return df, preproc_params

def trial_train(df):
    import statsmodels.formula.api as sm

    f = 'Default ~ A + B + C + D + E'
    
    model_ols = sm.ols(f, data=df).fit(disp=0)
    return(model_ols)

def trial_predict(test_df, model):
    predictions = model.predict(test_df)
    return(predictions)