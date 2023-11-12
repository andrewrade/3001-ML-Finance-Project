import pandas as pd
import argparse
import joblib

from utils import trial_preprocessor, predict_harness, trial_predict

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, help="enter some quality limit",
                    nargs='?', default='train.csv', const=0)
parser.add_argument("--output_csv", type=str, help="enter some quality limit",
                    nargs='?', default='train.csv', const=0)
args = parser.parse_args()

input_file = args.input_csv
output_file = args.output_csv

model_file = 'basic_model.sav'
model = joblib.load(model_file)
test = pd.read_csv(input_file)

test, preproc_params = trial_preprocessor(test)
actual_values, predictions, stats = predict_harness(test, model, trial_predict)
print(stats)

pd.DataFrame({
            "PD":list(predictions)
            }).to_csv(output_file, index=False)

# python3 harness.py --input_csv train.csv --output_csv predictions.csv 