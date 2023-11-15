from utils import stratified_split, predict_harness, plot_auc_rocs, get_roc, plot_roc_distribution
from tqdm import tqdm
import pandas as pd
import numpy as np

def bootstrapped_walk_forward_harness(df, preprocessor_function, train_function, predictor_function, start_index, step_size=1, num_bootstrap_samples = 1000):
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
    out_of_sample_stats_list = []
    test_roc_values = []
    out_of_sample_roc_values = []

    def bootstrap_sample(data, random_state):
        return data.sample(n=data.shape[0], replace=True, random_state=random_state, ignore_index=True)

    np.random.seed(42)

    for i in tqdm(range(num_bootstrap_samples)):
        test_truth = []
        test_predictions = []
        out_of_sample_truth = []
        out_of_sample_predictions = []
        random_state = i+1 # np.random.randint(1, 1000)

        bootstrap_data = bootstrap_sample(df, random_state=random_state)

        for k, step in enumerate(steps):    
            train = bootstrap_data[bootstrap_data[step_col] <= step]
            test = bootstrap_data[(bootstrap_data[step_col] >= step) & (bootstrap_data[step_col] < step + pd.DateOffset(years=1))]
            train, out_of_sample = stratified_split(train, label) 
            train.drop(columns=['id', 'stmt_date'], inplace=True)
            model = train_function(train)

            if test.shape[0]>0:
                actual_values, predictions, stats = predict_harness(test, model, predictor_function)
                test_stats_list.append(stats)
                test_truth += list(actual_values)
                test_predictions += list(predictions)
            else:
                print(k,n)
            
            # if out_of_sample.shape[0]>0:
            #     actual_values, predictions, stats = predict_harness(out_of_sample, model, predictor_function)
            #     out_of_sample_stats_list.append(stats)
            #     out_of_sample_truth += list(actual_values)
            #     out_of_sample_predictions += list(predictions)
            # else:
            #     print(k,n)
        
        test_roc_values.append(get_roc(test_truth, test_predictions))
        # out_of_sample_roc_values.append(get_roc(out_of_sample_truth, out_of_sample_predictions))
    plot_roc_distribution(test_roc_values)
    # plot_roc_distribution(out_of_sample_roc_values)

    model = train_function(df)
    return model, test_stats_list, out_of_sample_stats_list, preproc_params