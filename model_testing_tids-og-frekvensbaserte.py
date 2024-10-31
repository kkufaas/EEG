import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats
from scipy.signal import butter, filtfilt

# Paths for train and test data
train_folder = "/Users/kristina/Dropbox/Mac/Desktop/Train copy"
test_folder = "/Users/kristina/Dropbox/Mac/Desktop/Test"

columns_raw_data = [
    'trial_number', 'sensor_position', 'sample_num', 'sensor_value',
    'subject_identifier', 'matching_condition', 'channel', 'name', 'time'
]

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


# Load all trials from the specified folder
def load_all_trials_from_folder(folder_path):
    all_trials = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            trial_data = pd.read_csv(file_path, sep=',', header=0, names=columns_raw_data)
            all_trials.append(trial_data)
    return pd.concat(all_trials, ignore_index=True)


# Define feature extraction functions (no changes from original)
def split_into_windows(group, window_size=256):
    n_samples = len(group)
    windows = []
    for start_idx in range(0, n_samples, window_size):
        end_idx = start_idx + window_size
        if end_idx <= n_samples:
            windows.append(group.iloc[start_idx:end_idx])
    return windows


def calculate_time_domain_features(window):
    sensor_values = window['sensor_value'].values
    features = {
        'mean_amplitude': np.mean(sensor_values),
        'variance': np.var(sensor_values),
        'skewness': stats.skew(sensor_values),
        'kurtosis': stats.kurtosis(sensor_values),
        'max_value': np.max(sensor_values),
        'min_value': np.min(sensor_values),
        'rms': np.sqrt(np.mean(sensor_values ** 2))
    }
    return features


def extract_features_from_trials(trials, window_size=256):
    feature_list = []
    grouped = trials.groupby(['trial_number', 'sensor_position', 'matching_condition'])
    for name, group in grouped:
        subject_identifier = group['subject_identifier'].iloc[0]
        patient_id = group['name'].iloc[0]
        windows = split_into_windows(group, window_size)
        for window in windows:
            features = calculate_time_domain_features(window)
            features['trial_number'] = name[0]
            features['sensor_position'] = name[1]
            features['subject_identifier'] = subject_identifier
            features['matching_condition'] = group['matching_condition'].iloc[0]
            features['name'] = patient_id
            feature_list.append(features)
    return pd.DataFrame(feature_list)


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


def extract_frequency_features_from_trials(trials, window_size=256, sampling_rate=256):
    feature_list = []
    grouped = trials.groupby(['trial_number', 'sensor_position', 'matching_condition'])

    for name, group in grouped:
        subject_identifier = group['subject_identifier'].iloc[0]
        patient_id = group['name'].iloc[0]
        windows = split_into_windows(group, window_size)

        for window in windows:
            sensor_values = window['sensor_value'].values
            # Apply bandpass filter
            filtered_sensor_values = bandpass_filter(sensor_values, 0.5, 30, sampling_rate)
            # Categorize frequency bands
            freq_bands = categorize_frequency_bands(filtered_sensor_values, sampling_rate)

            # Add identifiers and append
            freq_bands['sensor_position'] = name[1]
            freq_bands['subject_identifier'] = subject_identifier
            freq_bands['matching_condition'] = group['matching_condition'].iloc[0]
            freq_bands['name'] = patient_id

            feature_list.append(freq_bands)

    return pd.DataFrame(feature_list)


if __name__ == "__main__":
    # Load and process train data
    train_data = load_all_trials_from_folder(train_folder)
    extracted_time_features_train_df = extract_features_from_trials(train_data)
    extracted_frequency_features_train_df = extract_frequency_features_from_trials(train_data)
    combined_features_train_df = pd.concat([extracted_time_features_train_df, extracted_frequency_features_train_df],
                                           axis=1)

    # Load and process test data
    test_data = load_all_trials_from_folder(test_folder)
    extracted_time_features_test_df = extract_features_from_trials(test_data)
    extracted_frequency_features_test_df = extract_frequency_features_from_trials(test_data)
    combined_features_test_df = pd.concat([extracted_time_features_test_df, extracted_frequency_features_test_df],
                                          axis=1)

    # Define feature columns and target
    feature_columns = ['mean_amplitude', 'variance', 'skewness', 'kurtosis', 'rms', 'min_value', 'max_value',
                       'delta', 'theta', 'alpha', 'beta', 'theta_alpha_ratio', 'delta_beta_ratio']
    X_train = combined_features_train_df[feature_columns]
    y_train = extracted_time_features_train_df['subject_identifier']
    X_test = combined_features_test_df[feature_columns]
    y_test = extracted_time_features_test_df['subject_identifier']

    # Define pipelines
    pipelines = {
        'SVM (Pipeline)': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('svc', SVC(probability=True, random_state=42, class_weight='balanced'))
        ]),
        'Decision Tree (Pipeline)': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('decision_tree', DecisionTreeClassifier(random_state=42))
        ]),
        'KNN (Pipeline)': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('knn', KNeighborsClassifier())
        ]),
        'Random Forest (Pipeline)': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('random_forest', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])
    }

    # Train and evaluate each model
    for model_name, pipeline in pipelines.items():
        print(f"\nTraining {model_name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"\n{model_name} Model")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
