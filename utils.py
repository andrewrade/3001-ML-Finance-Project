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

def get_probability(num, probability_map):
    return probability_map[num] if num in probability_map else probability_map[min(probability_map.keys(), key=lambda k: abs(k-num))]

def set_probability(df):
    probability_map = get_bins(df, 'predictions')
    return df['predictions'].apply(lambda x: get_probability(x, probability_map))

def get_bins(dataset, column):
    n = 200 #n_of_buckets #optimal_num_bins(dataset[column])
    dataset[column].fillna(0, inplace=True)
    # Calculate histogram
    hist, bin_edges = np.histogram(dataset[column], bins=dataset[column].quantile(np.linspace(0,1,n+1)))

    bins = []
    percentage = []
    for i in range(n):
        bins.append((bin_edges[i]+bin_edges[i+1])/2)
        mask = (bin_edges[i] <= dataset[column]) & (dataset[column] < bin_edges[i + 1])
        percentage.append(dataset['Default'][mask].mean())
    # print(bin_edges)
    percentage = [0 if pd.isna(x) else x for x in percentage]
    probability_map = {b:p for b,p in zip(bins, percentage)}
    return probability_map

def plot_defaults(dataset, column):
    probability_map = get_bins(dataset, column)
    bins = []
    percentage = []
    for b,p in probability_map.items():
        bins.append(b)
        percentage.append(p)

    plt.figure(figsize=(15, 6))
    plt.bar(bins, percentage, width=0.0002, color='skyblue')
    plt.xlabel(column)
    plt.xticks(rotation=45)
    plt.ylabel('Percentage of Default')
    plt.title(f'Percentage of Default VS {column}')
    plt.show()