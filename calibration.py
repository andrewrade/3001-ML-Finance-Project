import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import scipy.stats as st
import json

def population_pd_calibration(df, n_samples=10000):
    '''
        Estimates PD for the broader population via walk forward bootstrap. Bootstrap is performed independently for each year in data set
        and results combined at end in order to maintain temporal nature of underlying data 
    Parameters:
        df: Dataframe with columns 'fs_year' and 'Default' (binary class label)
        n_samples: Int, number of bootstrapped samples to produce 
        confidence: float, confidence level to use for interval (default is 95%)
    Returns:
        bootstrap_pd_stats: dictionary with results from bootstrapping
            mean: mean PD (%)
            std: standard deviation of PDs (%)
            std_error: standard error of the estimate of mean PD 
            99%_ci: Confidence interval for estimate of the mean PD 
    '''
    years  = df['fs_year'].unique()
    n_years = len(years)
    n_records = len(df)
    
    bootstrapped_results = np.empty(shape=(n_years, n_samples))

    for i, year in enumerate(years):
        single_year_defaults = df[df['fs_year'] == year]
        n_companies = len(single_year_defaults)
        # Bootstrap single year and sum number of defaults in each sample
        bootstrapped_samples = np.array([
                single_year_defaults.sample(n_companies, replace=True)['Default'].sum() 
                for _ in range(n_samples)
            ])
        bootstrapped_results[i, :] = bootstrapped_samples

    # Sum results from each year & divide by total num records to get overall PDs
    agg_boostrapped_results = np.sum(bootstrapped_results, axis=0) / n_records

    # PD bootstrap statistics 
    mean_pd = np.mean(agg_boostrapped_results)
    std_pd = np.std(agg_boostrapped_results, ddof=1)
    std_error_pd = std_pd / np.sqrt(n_records)
    ci = st.norm.interval(alpha=0.95, loc=mean_pd, scale=std_error_pd)

    print(f'mean default:{mean_pd:.2%}')
    print(f'default std:{std_pd:.2%}')

    bootstrapped_pd_stats ={
        'mean': mean_pd,
        'std': std_pd,
        'std_error': std_error_pd,
        '99%_ci': ci
    }
    
    return agg_boostrapped_results, bootstrapped_pd_stats


def optimal_num_bins(data):
    """
    Determines the number of bins based on the Freedman-Diaconis rule, 
    which is based on IQR instead of the number of datapoints
    Parameters:
        data: The data for which to calculate the number of bins.
    Returns:
        n_bins: The number of bins.
    """
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * IQR * (len(data) ** (-1/3))
    num_bins = int((np.max(data) - np.min(data)) / bin_width)
    return max(1, num_bins)  # Ensure at least one bin


def non_parametric_pd_calibration(y_true, preds):
    '''
        Adjust for model bias by calibrating binned PDs to the observed PD within the bins
    Parameters:  
        y_true: n x 1 array of ground truth labels
        preds: n x 1 array of PDs produced by the model 
    Returns:
        
    '''
    pd_and_label = np.column_stack((y_true, preds))
    sorted_pd_and_label = pd_and_label[pd_and_label[:, 1].argsort()]
    
    num_bins = optimal_num_bins(sorted_pd_and_label[:, 1])
    bins = np.linspace(np.min(sorted_pd_and_label[:, 1]), np.max(sorted_pd_and_label[:, 1]), num=num_bins)
    bin_indices = np.digitize(sorted_pd_and_label[:, 1], bins)

    pcts = []
    for i in range(1, len(bins)):
        bin_data = sorted_pd_and_label[bin_indices == i, 0]


def to_percent(y, _): #Format PD plots to have percentage on x-axis
    return "{:.2f}%".format(100 * y)

def main():
    df_processed = pd.read_csv("train_processed.csv")
    df_processed = df_processed[['fs_year', 'Default']]
    results, stats = population_pd_calibration(df_processed, n_samples=10000)

    # Plot bootstrapped sample histogram with 95% CI and Mean
    sns.set()
    plt.figure(figsize=(10,8))
    plt.hist(results, bins='auto', color='gray', alpha=0.7)
    formatter = FuncFormatter(to_percent)# Format PDs to be % on hist axis
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.axvline(x=stats['mean'], linestyle='dashed', color='black', label=f'Mean Population PD={stats["mean"]:.2%}')
    plt.axvline(x=stats['99%_ci'][0], linestyle='dashed', color='blue', label=f'99% Lower Bound={stats["99%_ci"][0]:.2%}')
    plt.axvline(x=stats['99%_ci'][1], linestyle='dashed', color='blue', label=f'99% Upper Bound={stats["99%_ci"][1]:.2%}')
    plt.title("Distribution of Boostraspped Population PD (10,000 samples)")
    plt.ylabel("Count")
    plt.xlabel("PD")
    plt.legend()
    plt.savefig('bootstrapped_PD.jpg', format='jpg', dpi=300)
    plt.show()
    
    output = 'bootstrapped_pd_stats.json'
    with open(output, 'w') as outfile:
        json.dump(stats, outfile, indent=4)

if __name__ == '__main__':
    main()

