from utils import stratified_split, get_roc, plot_roc_distribution, train_test_split_by_year
from tqdm import tqdm
import pandas as pd
import numpy as np
from estimate import estimation
from prediction import predict_harness

def bootstrapped_walk_forward_harness(df, preprocessor_function, preproc_params, train_function, start_index, step_size=1, num_bootstrap_samples = 10, model_type='Logit'):
    
    df = preprocessor_function(df, preproc_params, label=True, interest_rates=True, one_hot_encode=True)

    label='Default'
    step_col='stmt_date'
    start_index = pd.to_datetime(start_index)
    df[step_col] = pd.to_datetime(df[step_col])

    steps = [start_index]
    step = start_index+pd.DateOffset(years= step_size)
    while step<df[step_col].max():
        steps.append(step)
        step = step+pd.DateOffset(years= step_size)
    n = len(steps)
    steps = steps[2:]
    print(steps)

    test_truth = [[] for _ in range(num_bootstrap_samples)]
    test_predictions = [[] for _ in range(num_bootstrap_samples)]
    test_stats_list = [[] for _ in range(num_bootstrap_samples)]
    test_roc_values = []
    
    np.random.seed(42)

    for k, step in enumerate(tqdm(steps)):    
        train = df[df[step_col] <= step]
        test = df[(df[step_col] >= step) & (df[step_col] < step + pd.DateOffset(years=1))]
        
        df_train = train.copy()
        df_train.drop(columns='id', inplace=True)

        model = train_function(df_train = df_train, model_type = model_type)

        if test.shape[0]>0:

            for i in range(num_bootstrap_samples):

                bootstrap_test = test.sample(n=test.shape[0], replace=True, random_state=i+1, ignore_index=True)
                bootstrap_test.drop('id', axis=1, inplace=True)
                
                actual_values, predictions, stats = predict_harness(bootstrap_test, model, model_type)
                print(stats)
                test_stats_list[i].append(stats)
                test_truth[i] += list(actual_values)
                test_predictions[i] += list(predictions)

        else:
            print(k,n)

    for i in range(num_bootstrap_samples):
        test_roc_values.append(get_roc(test_truth[i], test_predictions[i]))
        
    plot_roc_distribution(test_roc_values, model_type)

    # Drop id from full df before passing to train
    train, test = train_test_split_by_year(df, date_column='stmt_date', test_frac=0.4)
    test.to_csv("csv_files/test.csv", index=False) # Save train data
    
    train.drop(columns='id', inplace=True)
    model = train_function(df_train = train, model_type = model_type)
    
    return model, test_stats_list, None