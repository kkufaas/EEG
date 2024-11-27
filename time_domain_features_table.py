import os

import pandas as pd
import numpy as np
from scipy import stats

pd.set_option('display.max_columns', None)
train_folder = "/Users/kristina/Dropbox/Mac/Desktop/Train copy"
test_folder = "/Users/kristina/Dropbox/Mac/Desktop/Test"


def load_and_preprocess_data(folder_path):
    columns = ['trial_number', 'sensor_position', 'sample_num', 'sensor_value', 'subject_identifier',
               'matching_condition', 'channel', 'name', 'time']
    all_trials = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            trial_data = pd.read_csv(file_path, sep=',', header=0, names=columns)
            all_trials.append(trial_data)
    return pd.concat(all_trials, ignore_index=True)


EEG_data = load_and_preprocess_data(train_folder)


# Define function to calculate time-domain features
def calculate_time_domain_features(sensor_values):
    return {
        'mean_amplitude': np.mean(sensor_values),
        'variance': np.var(sensor_values),
        'skewness': stats.skew(sensor_values),
        'kurtosis': stats.kurtosis(sensor_values),
        'max_value': np.max(sensor_values),
        'min_value': np.min(sensor_values),
        'rms': np.sqrt(np.mean(sensor_values ** 2))
    }


# Initialize an empty list to store the feature summary
time_domain_summary = []

# Group by `matching_condition` and `subject_identifier` and calculate time-domain features
for (matching_condition, subject_identifier), group in EEG_data.groupby(['matching_condition', 'subject_identifier']):
    # Calculate features for each group
    features = calculate_time_domain_features(group['sensor_value'].values)
    features['matching_condition'] = matching_condition
    features['subject_identifier'] = subject_identifier
    time_domain_summary.append(features)

# Convert the summary to a DataFrame
time_domain_summary_df = pd.DataFrame(time_domain_summary)

# Display the table
print(time_domain_summary_df)

# Save the table to a CSV file (optional)
time_domain_summary_df.to_csv("time_domain_features_summary.csv", index=False)
