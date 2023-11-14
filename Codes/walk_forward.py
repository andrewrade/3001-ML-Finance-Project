from utils import stratified_split, plot_auc_rocs, get_roc, plot_roc_distribution
from tqdm import tqdm
import pandas as pd
import numpy as np
import estimate
from prediction import predict_harness

def bootstrapped_walk_forward_harness(df, preprocessor_function, preproc_params, train_function, start_index, step_size=1, num_bootstrap_samples = 10, model_type='Logit'):
    
    df = preprocessor_function(df, preproc_params, label=True, interest_rates=True, one_hot_encode=True)

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
    steps = steps[2:]
    print(steps)

    test_stats_list = []
    out_of_sample_stats_list = []
    test_roc_values = []
    out_of_sample_roc_values = []

    def bootstrap_sample(data, random_state):
        return data.sample(n=data.shape[0], replace=True, random_state=random_state, ignore_index=True)

    np.random.seed(42)
    i=0

    for _ in tqdm(range(num_bootstrap_samples)):
        
        test_truth = []
        test_predictions = []
        out_of_sample_truth = []
        out_of_sample_predictions = []
        random_state = i
        i +=1
        print(i)

        bootstrap_data = bootstrap_sample(df, random_state=random_state)

        for k, step in enumerate(steps):    
            train = bootstrap_data[bootstrap_data[step_col] <= step]
            test = bootstrap_data[(bootstrap_data[step_col] >= step) & (bootstrap_data[step_col] < step + pd.DateOffset(years=1))]
            train, out_of_sample = stratified_split(train, label) 

            model = train_function(df_train = train, model_type = model_type)

            if test.shape[0]>0:
                actual_values, predictions, stats = predict_harness(test, model, model_type)
                print(stats)
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

    model = train_function(df_train = df, model_type = model_type)
    return model, test_stats_list, out_of_sample_stats_list