from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def train_test_split_by_year(df, date_column, test_frac):
    """
    Split passed df into train and test sets. Test is comprised of 
    test_frac % of the samples from the last year of data in test set.
    """
    df['year'] = df[date_column].dt.year # Extract year from 'stmt_date'
    last_year = df['year'].max()
    last_year_data = df[df['year'] == last_year]  # Separate out last year of data
    prev_years_train = df[df['year'] < last_year]
    
    # Split the last year's data into 50% test and 50% train
    last_year_train, test = train_test_split(last_year_data, test_size=test_frac, random_state=42)
    
    # Combine the previous years' data with the last year's training data
    train = pd.concat([prev_years_train, last_year_train])
    
    train = train.drop(columns=['year'])
    test = test.drop(columns=['year'])
    
    return train, test

def stratified_split(df, label):
    grouped_df = df.groupby('id').agg({'id': 'count', label: 'max'})
    grouped_df = grouped_df.rename(columns={'id': 'count', label: 'default_max'})
    grouped_df = grouped_df.reset_index()

    in_sample_companies, out_of_sample_companies = train_test_split(df, test_size=0.1, random_state=42)
    in_sample_companies = in_sample_companies['id'].values
    out_of_sample_companies = out_of_sample_companies['id'].values

    return df[df.id.isin(in_sample_companies)], df[df.id.isin(out_of_sample_companies)]

def plot_roc_distribution(roc_values, model_name, auc):
    plt.figure(figsize=(8, 8))

    for fpr, tpr, _ in roc_values:
        plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic, AUC = {:.2f})'.format(auc))
    plt.legend()
    plt.savefig(f'figs/{model_name}_roc_distribution.png', format='png')

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