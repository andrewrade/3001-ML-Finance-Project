# Instruction to run code

#### Final Harness -  python harness.py --input_csv '/Users/chitvangoyal/Desktop/3001_project/3001-ML-Finance-Project/train.csv' --output_csv 'pred.csv'

#### python train_harness.py to train and generate ROC auc curve (save train.csv in csv_files)

# Model File

#### Calibration.sav
#### xgb_model.sav

# Codes

#### estimate.py - returns trained models
#### preprocessor.py - returns processed dataframe for both test and train
#### predictions.py - returns predictions
#### walkforward.py - returns walk forward stats for AUC
#### utils.py - includes helper function
#### Harness.py - code for submission/ prediction on holout dataset
#### Calibration.py - code for non-parametric and baseline PD calibration


# Files

#### atc_industry_mapping - Consolidated sector code mapping for categorical encoding
#### ateco_sector_mapping - ateco sector code mapping for categorical encoding
#### legal struct_mapping - legal structure of firm mapping for categorical encoding
#### ECB Data Portal_20231029154614 - ECB historical interest rate data

# Notebook

#### Explain.ipynb - for explainability
#### Z_score_and_plots - for Z score calculation



