import tarfile
import pandas as pd
import numpy as np
import os

import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

###################################### README ########################################
# Add path to your local folder here. GitHub has a file size limit of 100 MB per file.
# Files can be downloaded here> https://www.kaggle.com/datasets/nnair25/Alcoholics/data
# There are Test and Train folders, and loose files.
# Only keep Train folder on your PC, it's enough for our task.
train_folder = "/Users/kristina/Dropbox/Mac/Desktop/Train"
test_folder = "/Users/kristina/Dropbox/Mac/Desktop/Test"
######################################################################################


columns_raw_data = [
    'trial_number', 'sensor_position', 'sample_num', 'sensor_value',
    'subject_identifier', 'matching_condition', 'channel', 'name', 'time'
]

weights = {
    'mean_amplitude': 0.05,
    'variance': 0.20,
    'skewness': 0.05,
    'kurtosis': 0.05,
    'max_value': 0.10,
    'min_value': 0.15,
    'rms': 0.40
}

sampling_rate = 256.0  # in Hz. It means that there are 256 samples are recorded every second for each EEG sensor.
# The sensor values in dataset (in micro volts) are processed with FFT (see apply_fft) to compute the power in different
# frequency bands. With 256 Hz, the highest frequency we can analyze (according to the Nyquist theorem) is half of the
# sampling rate,ni.e., 128 Hz. This is more than enough to analyze the brainwave frequencies, which go up to 30 Hz
# (beta band).

# DEBUG: sets pandas options to display all columns
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.expand_frame_repr', False)


def load_all_trials_from_folder(folder_path):
    """ Load all CSV files from the specified folder and return a concatenated DataFrame.
    Params: folder_path (str): The path to the folder containing CSV files.
    Returns: pd.DataFrame: A DataFrame containing the concatenated data from all CSV files."""
    all_trials = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            trial_data = pd.read_csv(file_path, sep=',', header=0, names=columns_raw_data)
            all_trials.append(trial_data)
    return pd.concat(all_trials, ignore_index=True)


def split_into_windows(group, window_size=256):
    """ Splits a group of sensor data into windows of a fixed size (here, 256 samples per window).
    Params: group (pd.DataFrame): A group of sensor values for a specific trial and sensor.
            window_size (int): The number of samples per window
    Returns: list: A list of windows (each window is a DataFrame)"""
    n_samples = len(group)
    windows = []
    for start_idx in range(0, n_samples, window_size):
        end_idx = start_idx + window_size
        if end_idx <= n_samples:
            windows.append(group.iloc[start_idx:end_idx])
    return windows


def calculate_time_domain_features(window):
    """
    Calculate time-domain statistical features from an EEG signal window.
    Params:
        window (pd.DataFrame): A DataFrame containing the sensor values in a single window.
    Returns:
        dict: A dictionary containing the extracted features.
    """
    sensor_values = window['sensor_value'].values

    features = {
        # Mean Amplitude
        'mean_amplitude': np.mean(sensor_values),

        # StDev
        'standard deviation': np.std(sensor_values),

        # Variance
        'variance': np.var(sensor_values),

        # Skewness
        'skewness': stats.skew(sensor_values),

        # Kurtosis
        'kurtosis': stats.kurtosis(sensor_values),

        # Maximum Value
        'max_value': np.max(sensor_values),

        # Minimum Value
        'min_value': np.min(sensor_values),

        # Root Mean Square (RMS)
        'rms': np.sqrt(np.mean(sensor_values ** 2))
    }
    return features


def extract_features_from_trials(trials, window_size=256):
    """
    Extract statistical features from all windows in the provided trials.
    Params:
        trials (pd.DataFrame): A DataFrame containing all EEG trials data.
        window_size (int): Number of samples per window.
    Returns:
        pd.DataFrame: A DataFrame containing the extracted features for each window.
    """
    feature_list = []
    grouped = trials.groupby(['trial_number', 'sensor_position', 'matching_condition'])
    for name, group in grouped:
        subject_identifier = group['subject_identifier'].iloc[0]
        patient_id = group['name'].iloc[0]
        windows = split_into_windows(group, window_size)

        for window in windows:
            features = calculate_time_domain_features(window)

            # features['trial_number'] = name[0]
            features['sensor_position'] = name[1]
            features['subject_identifier'] = subject_identifier  # Add subject identifier
            features['matching_condition'] = group['matching_condition'].iloc[0]
            features['name'] = patient_id

            feature_list.append(features)
    return pd.DataFrame(feature_list)


def apply_fft(data, sampling_rate):
    """ Group the data by trial and sensor, filter the signal, and apply FFT.
    Params: data (pd.DataFrame): The dataset containing EEG trials and sensor readings.
            sampling_rate (float): The sampling rate of the EEG signal (in Hz).
    Returns: list: A list of tuples containing trial/sensor identifiers and their corresponding frequency bands."""

    freq_bands_list = []
    grouped = data.groupby(['trial_number', 'sensor_position'])

    for name, group in grouped:
        windows = split_into_windows(group)
        for window in windows:
            if len(window) == 256:
                sensor_values = window['sensor_value'].values

                # Step 1: Apply the band-pass filter to clean the signal (0.5 to 30 Hz). EEG signals
                # often contain noise. The band-pass filter removes frequencies outside the 0.5 to 30 Hz range,
                # which are relevant EEG frequency bands (delta, theta, alpha, beta).
                filtered_sensor_values = bandpass_filter(sensor_values, 0.5, 30, sampling_rate)

                # Step 2: Apply the FFT (the Fast Fourier Transform) and categorize the frequencies.
                # https://en.wikipedia.org/wiki/Fast_Fourier_transform
                # FFT is aimed to convert the time-domain signal into the frequency domain.
                # This allows us to compute the power in each frequency band and then categorize them.
                freq_bands = categorize_frequency_bands(filtered_sensor_values, sampling_rate)
                freq_bands_list.append((name, freq_bands))
            else:
                print(f"Skipping window for trial {name[0]} sensor {name[1]} due to insufficient data points.")
    return freq_bands_list if freq_bands_list else None


def categorize_frequency_bands(sensor_values, sampling_rate):
    """ Converts time-domain EEG sensor values into frequency domain and categorizes into
    alpha, beta, delta, and theta bands.
    Params: sensor_values (array-like): Time-series sensor values in micro volts.
            sampling_rate (float): The sampling rate of the signal (in Hz).
    Returns: freq_bands (dict): A dictionary with power in alpha, beta, delta, and theta bands.
    Source: https://en.wikipedia.org/wiki/Electroencephalography"""

    # Step 1: Calculate the number of samples in the time-series data
    n_samples = len(sensor_values)

    # Step 2: Apply FFT to convert the time-domain signal to the frequency domain
    # FFT decomposes the time-series signal into its frequency components. It helps in understanding the contribution
    # of different frequencies (brainwave bands) to the overall signal. The result is a set of complex numbers that
    # represent the amplitude and phase of various frequency components.
    fft_values = np.fft.fft(sensor_values)

    # Step 3: Calculate the corresponding frequency values for each FFT result
    # 'np.fft.fftfreq' returns the frequencies associated with each FFT value. The frequency range spans from 0 Hz
    # up to the Nyquist frequency (half the sampling rate), which is 128 Hz in this case (256 Hz / 2).
    freqs = np.fft.fftfreq(n_samples, 1 / sampling_rate)

    # Step 4: Compute the power spectrum, which is the magnitude of the FFT values squared
    # The power spectrum tells us how much "power" or "energy" exists at each frequency and
    # reveals which frequency bands (delta, theta, alpha, beta) are most active.
    power_spectrum = np.abs(fft_values) ** 2

    # Only keep positive frequencies, because the FFT result is symmetric, which means it contains both positive and
    # negative frequencies The 'n_samples // 2' step ensures we're only considering frequencies from 0 to 128 Hz.
    pos_freqs = freqs[:n_samples // 2]
    pos_power_spectrum = power_spectrum[:n_samples // 2]

    # Step 7: Calculate the total power for each frequency band by summing the power spectrum
    # within specific frequency ranges
    freq_bands = {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0}

    # Delta band (0.5–4 Hz): Associated with deep sleep and unconscious states.
    delta_mask = (pos_freqs >= 0.5) & (pos_freqs < 4)
    freq_bands['delta'] = np.sum(pos_power_spectrum[delta_mask])

    # Theta band (4–7 Hz): Linked to drowsiness, relaxation, and light sleep.
    theta_mask = (pos_freqs >= 4) & (pos_freqs < 7)
    freq_bands['theta'] = np.sum(pos_power_spectrum[theta_mask])

    # Alpha band (8–13 Hz): Often observed during relaxed wakefulness, meditative states, or closed eyes.
    alpha_mask = (pos_freqs >= 8) & (pos_freqs < 13)
    freq_bands['alpha'] = np.sum(pos_power_spectrum[alpha_mask])

    # Beta band (13–30 Hz): Associated with active thinking, focus, and problem-solving.
    beta_mask = (pos_freqs >= 13) & (pos_freqs < 30)
    freq_bands['beta'] = np.sum(pos_power_spectrum[beta_mask])
    return freq_bands


def bandpass_filter(data, lowcut, highcut, sampling_rate, order=4):
    """
    Helper function. Applies the band-pass filter to clean the signal (0.5 to 30 Hz).
    EEG signals often contain noise. The band-pass filter removes frequencies outside the
    0.5 to 30 Hz range, which are relevant EEG frequency bands (delta, theta, alpha, beta).
    """
    # Nyquist's frequency is half the sampling rate
    nyquist = 0.5 * sampling_rate
    # Normalize the frequencies by the Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    # Use the butterworth filter, ensuring that two values are returned (b and a)
    b, a = butter(order, [low, high], btype='band')
    # Apply the filter using filtfilt (zero-phase filtering)
    y = filtfilt(b, a, data)
    return y


def calculate_power_distribution(data, sampling_rate, filename):
    """ Calculates the power distribution across delta, theta, alpha, and beta categories for all sensors in the data.
    Includes the name of the file and subject identifier from the dataset.
    Params:
        data (pd.DataFrame): The EEG dataset containing sensor readings.
        sampling_rate (float): The sampling rate of the signal (in Hz).
        filename (str): The name of the file from which the data was loaded.

    Returns: pd.DataFrame: A DataFrame summarizing the power in each frequency band for each sensor.
    """
    power_data = []
    grouped = data.groupby(['trial_number', 'sensor_position'])
    for name, group in grouped:
        sensor_values = group['sensor_value'].values
        freq_bands = categorize_frequency_bands(sensor_values, sampling_rate)
        freq_bands['trial_number'] = name[0]
        freq_bands['sensor_position'] = name[1]
        freq_bands['filename'] = filename
        freq_bands['subject_identifier'] = group['subject_identifier'].iloc[0]
        power_data.append(freq_bands)
    return pd.DataFrame(power_data)


def summarize_power_by_group(data):
    """ Summarize the average power in each frequency band by subject group (a or c)."""
    summary = data.groupby('subject_identifier')[['delta', 'theta', 'alpha', 'beta']].mean().reset_index()
    return summary


def conduct_t_tests(data):
    """Conduct independent t-tests to compare power in each frequency band between alcoholic and control groups."""
    ttest_results = {}
    alcoholic_data = data[data['subject_identifier'] == 'a']
    control_data = data[data['subject_identifier'] == 'c']
    for band in ['delta', 'theta', 'alpha', 'beta']:
        t_stat, p_value = stats.ttest_ind(alcoholic_data[band], control_data[band], equal_var=False)
        ttest_results[band] = {'t_stat': t_stat, 'p_value': p_value}
    return ttest_results

if __name__ == "__main__":
    columns = ['trial_number', 'sensor_position', 'sample_num', 'sensor_value', 'subject_identifier',
               'matching_condition', 'channel', 'name', 'time']

    train_data = load_all_trials_from_folder(train_folder)
    extracted_features_df = extract_features_from_trials(train_data)
    numeric_cols = extracted_features_df.select_dtypes(include=np.number).columns
    numeric_aggregated_features = extracted_features_df.groupby(['name', 'matching_condition'])[
        numeric_cols].mean().reset_index()

    non_numeric_cols = ['name', 'subject_identifier']
    non_numeric_data = extracted_features_df[non_numeric_cols].drop_duplicates('name')
    patient_aggregated_features = pd.merge(numeric_aggregated_features, non_numeric_data, on='name')

    # Calculate the cognitive score and add it as a new column
    patient_aggregated_features['cognitive_score'] = (
            patient_aggregated_features['mean_amplitude'] * weights['mean_amplitude'] +
            patient_aggregated_features['variance'] * weights['variance'] +
            patient_aggregated_features['skewness'] * weights['skewness'] +
            patient_aggregated_features['kurtosis'] * weights['kurtosis'] +
            patient_aggregated_features['max_value'] * weights['max_value'] +
            patient_aggregated_features['min_value'] * weights['min_value'] +
            patient_aggregated_features['rms'] * weights['rms']
    )

    alcoholic_data = patient_aggregated_features[patient_aggregated_features['subject_identifier'] == 'a']
    control_data = patient_aggregated_features[patient_aggregated_features['subject_identifier'] == 'c']

    # List of parameters to analyze
    parameters = ['mean_amplitude', 'variance', 'skewness', 'kurtosis', 'rms', 'min_value', 'max_value']
    matching_conditions = ['S1 obj', 'S2 match', 'S2 nomatch,']  # List of matching conditions to analyze

    # Initialize lists to store t-test results
    t_stats = []
    p_values = []
    matching_condition_list = []

    for condition in matching_conditions:
        for param in parameters:
            data_alcoholic = alcoholic_data[alcoholic_data['matching_condition'] == condition][param]
            data_control = control_data[control_data['matching_condition'] == condition][param]
            t_stat, p_value = ttest_ind(data_alcoholic, data_control)
            t_stats.append(t_stat)
            p_values.append(p_value)
            matching_condition_list.append(condition)

    t_test_summary = pd.DataFrame({
        'Parameter': parameters * len(matching_conditions),
        'Matching Condition': matching_condition_list,
        'T-statistic': t_stats,
        'P-value': p_values
    })

    print("\nT-Test:")
    print(t_test_summary)

    print("\nCognitive scores per patient for each Mmtching condition:")
    cognitive_scores_df = patient_aggregated_features[['name', 'matching_condition', 'cognitive_score']]
    print(cognitive_scores_df.to_string(index=False))  # Display without index


    # Step 2: Extract frequency-domain features from the trials data using FFT
    print("Extracting frequency-domain features from EEG signals using FFT...")
    frequency_features_df = apply_fft_with_features(train_data, sampling_rate)

    # Display the frequency-domain features for each trial
    print("\nFrequency-Domain Features (Head of DataFrame):")
    print(frequency_features_df.head())

    # Step 3: Summarize the power ratios and peak frequencies
    print("\nSummarized Power Ratios and Peak Frequencies:")
    print(frequency_features_df[['trial_number', 'sensor_position', 'peak_frequency', 'theta_alpha_ratio', 'delta_beta_ratio']].head())

    print("Feature extraction complete!")
