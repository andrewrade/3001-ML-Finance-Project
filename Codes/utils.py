from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prediction import predict_function

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

    in_sample_companies, out_of_sample_companies = train_test_split(df, test_size=0.1, random_state=42)
    in_sample_companies = in_sample_companies['id'].values
    out_of_sample_companies = out_of_sample_companies['id'].values

    return df[df.id.isin(in_sample_companies)], df[df.id.isin(out_of_sample_companies)]

def predict_harness(test, model, model_type):
    predictions = predict_function(test_df = test, model= model, model_type = model_type)

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
    plt.savefig('roc_distribution.png', format='png')

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
    plt.savefig('AUC_ROC.png', format='png')
