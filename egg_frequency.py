import pandas as pd
import numpy as np
import os

###################################### README ########################################
# Add path to your local folder here. GitHub has a file size limit of 100 MB per file.
# Files can be downloaded here> https://www.kaggle.com/datasets/nnair25/Alcoholics/data
train_folder = "/Users/kristina/Dropbox/Mac/Desktop/Train copy"
######################################################################################

sampling_rate = 256.0  # in Hz

columns = [
    'trial_number', 'sensor_position', 'sample_num', 'sensor_value',
    'subject_identifier', 'matching_condition', 'channel', 'name', 'time'
]

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def load_all_trials_from_folder(folder_path):
    """ Load all CSV files from the specified folder and return a concatenated DataFrame."""
    all_trials = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            trial_data = pd.read_csv(file_path, sep=',', header=0, names=columns)
            all_trials.append(trial_data)
    return pd.concat(all_trials, ignore_index=True)


def split_into_windows(group, window_size=256):
    """ Splits sensor data into windows of a fixed size (256 samples per window)."""
    n_samples = len(group)
    windows = []
    for start_idx in range(0, n_samples, window_size):
        end_idx = start_idx + window_size
        if end_idx <= n_samples:
            windows.append(group.iloc[start_idx:end_idx])
    return windows


def categorize_frequency_bands(sensor_values, sampling_rate):
    """ Converts time-domain EEG sensor values into frequency domain and categorizes into alpha, beta, delta,
    and theta bands. """
    n_samples = len(sensor_values)
    fft_values = np.fft.fft(sensor_values)
    freqs = np.fft.fftfreq(n_samples, 1 / sampling_rate)
    power_spectrum = np.abs(fft_values) ** 2
    pos_freqs = freqs[:n_samples // 2]
    pos_power_spectrum = power_spectrum[:n_samples // 2]

    freq_bands = {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0}
    delta_mask = (pos_freqs >= 0.5) & (pos_freqs < 4)
    theta_mask = (pos_freqs >= 4) & (pos_freqs < 7)
    alpha_mask = (pos_freqs >= 8) & (pos_freqs < 13)
    beta_mask = (pos_freqs >= 13) & (pos_freqs < 30)

    freq_bands['delta'] = np.sum(pos_power_spectrum[delta_mask])
    freq_bands['theta'] = np.sum(pos_power_spectrum[theta_mask])
    freq_bands['alpha'] = np.sum(pos_power_spectrum[alpha_mask])
    freq_bands['beta'] = np.sum(pos_power_spectrum[beta_mask])

    theta_alpha_ratio = freq_bands['theta'] / freq_bands['alpha'] if freq_bands['alpha'] > 0 else np.nan
    delta_beta_ratio = freq_bands['delta'] / freq_bands['beta'] if freq_bands['beta'] > 0 else np.nan

    freq_bands['theta_alpha_ratio'] = theta_alpha_ratio
    freq_bands['delta_beta_ratio'] = delta_beta_ratio
    return freq_bands


def extract_frequency_features_from_trials(trials, window_size=256):
    """ Extract frequency-domain features from all windows in the provided trials using FFT."""
    feature_list = []
    grouped = trials.groupby(['trial_number', 'sensor_position', 'matching_condition'])
    for name, group in grouped:
        subject_identifier = group['subject_identifier'].iloc[0]
        patient_id = group['name'].iloc[0]
        windows = split_into_windows(group, window_size)

        for window in windows:
            freq_bands = categorize_frequency_bands(window['sensor_value'].values, sampling_rate)
            freq_bands['sensor_position'] = name[1]
            freq_bands['subject_identifier'] = subject_identifier
            freq_bands['matching_condition'] = group['matching_condition'].iloc[0]
            freq_bands['name'] = patient_id
            feature_list.append(freq_bands)
    return pd.DataFrame(feature_list)


if __name__ == "__main__":
    # Step 1: Load all trials from the training folder
    train_data = load_all_trials_from_folder(train_folder)

    # Step 2: Extract frequency-domain features
    print("Extracting frequency-domain features from EEG signals...")
    extracted_frequency_features_df = extract_frequency_features_from_trials(train_data)

    # Step 3: Select only numeric columns for aggregation (skip non-numeric columns)
    numeric_cols = ['delta', 'theta', 'alpha', 'beta', 'theta_alpha_ratio', 'delta_beta_ratio']

    # Aggregate numeric features by patient and matching condition
    numeric_aggregated_features = extracted_frequency_features_df.groupby(
        ['name', 'matching_condition'])[numeric_cols].mean().reset_index()

    # Extract non-numeric columns to preserve information (e.g., subject_identifier)
    non_numeric_cols = ['name', 'subject_identifier']
    non_numeric_data = extracted_frequency_features_df[non_numeric_cols].drop_duplicates('name')

    # Merge aggregated numeric features with non-numeric data
    patient_aggregated_features = pd.merge(numeric_aggregated_features, non_numeric_data, on='name')

    # Step 4: Group by subject_identifier and matching_condition
    grouped_data = patient_aggregated_features.groupby(
        ['subject_identifier', 'matching_condition'])[numeric_cols].mean().reset_index()

    # Display the grouped data
    print("\n--- Grouped Data (Frequency-Domain Features) ---")
    print(grouped_data)

    print("Feature extraction and aggregation complete!")

