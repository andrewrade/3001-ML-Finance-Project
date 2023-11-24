import json
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


class SigmoidCalibrator:
    def __init__(self, prob_pred, prob_true):
        prob_pred, prob_true = self._filter_out_of_domain(prob_pred, prob_true)
        prob_true = np.log(prob_true / (1 - prob_true))
        self.regressor = LinearRegression().fit(
            prob_pred.reshape(-1, 1), prob_true.reshape(-1, 1)
        )

    def calibrate(self, probabilities):
        return 1 / (1 + np.exp(-self.regressor.predict(probabilities).flatten()))

    def _filter_out_of_domain(self, prob_pred, prob_true):
        filtered = list(zip(*[p for p in zip(prob_pred, prob_true) if 0 < p[1] < 1]))
        return np.array(filtered)


class IsotonicCalibrator:
    def __init__(self, prob_pred, prob_true):
        self.regressor = IsotonicRegression(out_of_bounds="clip")
        self.regressor.fit(prob_pred, prob_true)

    def calibrate(self, probabilities):
        return self.regressor.predict(probabilities)

def bayesian_pd_adjustment(pd_i, pd_sample, pd_true):
    '''
        Calibrate model output probabilities to reflect the ground truth population PD. 
        Adjustment factor computed via the Bayesian Method
    Parameters:
        pd_i: model probability output to be adjusted
        pd_sample: the sample population PD estimated by the model
        pd_true: the true population PD
    Returns:
        adj_pd_i: adjusted model probability
    '''
    adj_pd_i = pd_true * ((pd_i * (1-pd_sample)) / (pd_sample - pd_i * pd_sample + pd_i*pd_true - pd_sample*pd_true))
    return adj_pd_i


def bootstrap_population_pd(df, n_samples=10000):
    '''
        Estimates PD for the broader population via walk forward bootstrap. Bootstrap is performed independently for each year in data set
        and results combined at end in order to maintain temporal nature of underlying data 
    Parameters:
        df: Dataframe with columns 'fs_year' and 'Default' (binary class label)
        n_samples: Int, number of bootstrapped samples to produce 
        confidence: float, confidence level to use for interval (default is 99%)
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
    agg_bootstrapped_results = np.sum(bootstrapped_results, axis=0) / n_records

    # PD bootstrap statistics 
    mean_pd = np.mean(agg_bootstrapped_results)
    std_pd = np.std(agg_bootstrapped_results, ddof=1)
    std_error_pd = std_pd / np.sqrt(n_records)
    ci = st.norm.interval(alpha=0.99, loc=mean_pd, scale=std_error_pd)

    print(f'mean default:{mean_pd:.2%}')
    print(f'default std:{std_pd:.2%}')

    bootstrapped_pd_stats ={
        'mean': mean_pd,
        'std': std_pd,
        'std_error': std_error_pd,
        '99%_ci': ci
    }
    
    return agg_bootstrapped_results, bootstrapped_pd_stats


def non_parametric_pd_calibration(y_true, preds):
    '''
        Adjust for model bias by calibrating binned PDs to the observed PD within the bins
    Parameters:  
        y_true: n x 1 array of ground truth labels
        preds: n x 1 array of PDs produced by the model 
    Returns:
        calibration_curve_params: Parameters a,b,c of the fit curve a*exp(-b*x) + c
        bin_centers: discrete model PDs X used to fit calibration curve
        bin_pds: discrete true PDs y used to fit calibration curve
        
    '''
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

    def exponential_model(x, a, b, c):
        '''
        Exponential model to pass to curve_fit method 
        '''
        return a * np.exp(b*x) + c

    # Sort by model output PD and find bin widths
    pd_and_label = np.column_stack((y_true, preds))
    sorted_pd_and_label = pd_and_label[pd_and_label[:, 1].argsort()] 
    num_bins = optimal_num_bins(sorted_pd_and_label[:, 1])
    bins = np.linspace(np.min(sorted_pd_and_label[:, 1]), np.max(sorted_pd_and_label[:, 1]), num=num_bins)
    bin_indices = np.digitize(sorted_pd_and_label[:, 1], bins)

    # Compute PD for each bin 
    bin_pds = []
    for i in range(1, len(bins)):
        bin_data = sorted_pd_and_label[bin_indices == i, 0]
        if len(bin_data) > 0:
            bin_pd = np.sum(bin_data) / len(bin_data)
        else:
            bin_pd = 0 
        bin_pds.append(bin_pd)

    # Fit exponential curve to the data
    bin_centers = 0.5 * (bins[:-1] + bins[1:]) # Vectorize (a + b) / 2 for all bins 
    calibration_curve_params, _ = curve_fit(exponential_model, bin_centers, bin_pds)

    return calibration_curve_params, bin_centers, bin_pds


def main():

    def to_percent(y, _): #Format PD plots to have percentage on x-axis
        return "{:.2f}%".format(100 * y)
    
    df_processed = pd.read_csv("train_processed.csv")
    df_processed = df_processed[['fs_year', 'Default']]
    results, stats = bootstrap_population_pd(df_processed, n_samples=10000)

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

