from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def get_roc(actual_values, predictions):
    new_prediction_df = pd.DataFrame({
        'Actual Values': actual_values,
        'Predictions': predictions
    }).replace([np.inf, -np.inf], np.nan).dropna()

    actual_values =  new_prediction_df['Actual Values']
    predicted_probabilities =  new_prediction_df['Predictions']

    auc = roc_auc_score(actual_values, predicted_probabilities)
    fpr, tpr, thresholds = roc_curve(actual_values, predicted_probabilities)
    return [fpr, tpr, auc]

def plot_roc_distribution(roc_values):
    auc_values = [roc_value[2] for roc_value in roc_values]
    auc = sum(auc_values)/len(auc_values)
    plt.figure(figsize=(8, 8))

    for fpr, tpr, _ in roc_values:
        plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic, Average AUC = {:.2f})'.format(auc))
    plt.legend()

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