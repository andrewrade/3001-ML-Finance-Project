import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import scipy.stats as st
import json
import time

def bootstrap_pd(df, n_samples=10000):
    '''
        Estimates population default rate via walk forward bootstrap. Bootstrap is performed independently for each year in data set
        and results combined at end in order to maintain temporal nature of underlying data 

    Parameters:
        df: Dataframe with columns 'fs_year' and 'Default' (binary class label)
        n_samples: Int, number of bootstrapped samples to produce 
    Returns:
        bootstrap_pd_stats: dictionary with results from bootstrapping
            mean: mean PD (%)
            std: standard deviation of PD (%)
            95%_ci: 95% confidence interval 
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
    ci = st.norm.interval(alpha=0.95, loc=mean_pd, scale=std_pd)

    print(f'mean default:{mean_pd:.2%}')
    print(f'default std:{std_pd:.2%}')

    bootstrapped_pd_stats ={
        'mean': mean_pd,
        'std': std_pd,
        '95%_ci': ci
    }
    
    return agg_boostrapped_results, bootstrapped_pd_stats

def to_percent(y, _):
    return "{:.2f}%".format(100 * y)

def main():
    df_processed = pd.read_csv("train_processed.csv")
    df_processed = df_processed[['fs_year', 'Default']]
    results, stats = bootstrap_pd(df_processed, n_samples=10000)

    # Plot bootstrapped sample histogram with 95% CI and Mean
    sns.set()
    plt.figure(figsize=(10,8))
    plt.hist(results, bins='auto', color='gray', alpha=0.7)
    formatter = FuncFormatter(to_percent)# Format PDs to be % on hist axis
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.axvline(x=stats['mean'], linestyle='dashed', color='black', label=f'Mean PD={stats["mean"]:.2%}')
    plt.axvline(x=stats['95%_ci'][0], linestyle='dashed', color='blue', label=f'95% CI Lower Bound={stats["95%_ci"][0]:.2%}')
    plt.axvline(x=stats['95%_ci'][1], linestyle='dashed', color='blue', label=f'95% CI Upper Bound={stats["95%_ci"][1]:.2%}')
    plt.title("Bootsrapped PD Distribution (10,000 samples)")
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

