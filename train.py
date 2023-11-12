import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib

from walk_forward_utils import bootstrapped_walk_forward_harness

from utils import trial_preprocessor, trial_train, trial_predict
df = pd.read_csv('train.csv')
df.head()
start_index = df['stmt_date'].min()
model, test_stats_list, out_of_sample_stats_list, preproc_params = bootstrapped_walk_forward_harness(df, trial_preprocessor, trial_train, trial_predict, start_index, step_size=1, num_bootstrap_samples=1000)

filename = 'basic_model.sav'
joblib.dump(model, filename)