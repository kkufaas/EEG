import numpy as np
from scipy.stats import mannwhitneyu
import os
import pywt
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

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


# print("Data loaded and preprocessed. Sample data:")
# print(EEG_data.head())
# print("Column names in EEG_data:", EEG_data.columns)
# print("Unique values in 'subject_identifier' column:")
# print(EEG_data['subject_identifier'].unique())

def preprocess_data(X_train, y_train, X_test, balance_data=True):
    """
    Preprocess data with scaling and optional class balancing.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target labels.
        X_test (pd.DataFrame): Test features.
        balance_data (bool): Whether to balance classes using SMOTE.

    Returns:
        tuple: Processed X_train, y_train, X_test.
    """
    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balance classes
    if balance_data:
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print("Class distribution after SMOTE:", pd.Series(y_train).value_counts())
    return X_train_scaled, y_train, X_test_scaled


def extract_features_from_trials(trials, window_size=256, sensor_pairs=None, significant_channels=None,
                                 sampling_rate=256, wavelet=None, wavelet_level=4):
    feature_list = []
    grouped = trials.groupby(['trial_number', 'sensor_position', 'matching_condition'])
    print(f"Processing {len(grouped)} groups for feature extraction...")
    for name, group in tqdm(grouped, desc="Extracting Features"):
        subject_identifier = group['subject_identifier'].iloc[0]
        patient_id = group['name'].iloc[0]
        windows = split_into_windows(group, window_size)
        if len(windows) == 0:
            print(f"No windows generated for trial: {name}, class: {subject_identifier}")
            continue
        for window in windows:
            features = calculate_time_domain_features(window)
            features.update(calculate_frequency_domain_features(window, sampling_rate))
            if wavelet:
                features.update(calculate_wavelet_features(window, wavelet, wavelet_level))
            if sensor_pairs:
                features.update(calculate_correlation_features(window, sensor_pairs))
            if significant_channels:
                features.update(calculate_region_average_features(window, significant_channels))
            features['trial_number'] = name[0]
            features['sensor_position'] = name[1]
            features['subject_identifier'] = subject_identifier
            features['matching_condition'] = group['matching_condition'].iloc[0]
            features['name'] = patient_id
            feature_list.append(features)
    result_df = pd.DataFrame(feature_list)
    return result_df


def get_correlated_pairs(stimulus, threshold, group):
    trial_numbers_list = EEG_data['trial_number'][
        (EEG_data['subject_identifier'] == group) &
        (EEG_data['matching_condition'] == stimulus)
        ].unique()

    sample_correlation_df = pd.pivot_table(
        EEG_data[(EEG_data['subject_identifier'] == group) & (EEG_data['trial_number'] == trial_numbers_list[0])],
        values='sensor_value', index='sample_num', columns='sensor_position'
    ).corr()
    list_of_pairs = [f"{col}-{sample_correlation_df.index[i]}"
                     for j, col in enumerate(sample_correlation_df.columns)
                     for i in range(j + 1, len(sample_correlation_df))]

    corr_pairs_dict = {pair: 0 for pair in list_of_pairs}

    for trial_number in trial_numbers_list:
        correlation_df = pd.pivot_table(
            EEG_data[(EEG_data['subject_identifier'] == group) & (EEG_data['trial_number'] == trial_number)],
            values='sensor_value', index='sample_num', columns='sensor_position'
        ).corr()

        for j, column in enumerate(correlation_df.columns):
            for i in range(j + 1, len(correlation_df)):
                if correlation_df.iloc[i, j] >= threshold:
                    pair_name = f"{column}-{correlation_df.index[i]}"
                    if pair_name in corr_pairs_dict:
                        corr_pairs_dict[pair_name] += 1

    corr_count = pd.DataFrame(list(corr_pairs_dict.items()), columns=['channel_pair', 'count'])
    corr_count['group'] = group
    corr_count['stimulus'] = stimulus
    corr_count['ratio'] = corr_count['count'] / len(trial_numbers_list)
    return corr_count


def calculate_wavelet_features(window, wavelet='db4', level=4):
    """
    Calculate wavelet transform features for a given EEG window.
    Args:
        window (pd.DataFrame): A DataFrame containing 'sensor_value' for a single EEG window.
        wavelet (str): Wavelet type to use. Common options include 'db4', 'coif1', 'sym5', etc.
        level (int): Number of decomposition levels for the wavelet transform.

    Returns:
        dict: Dictionary of wavelet energy features for each frequency band.
    """
    sensor_values = window['sensor_value'].values
    coeffs = pywt.wavedec(sensor_values, wavelet, level=level)
    wavelet_features = {}
    for i, coeff in enumerate(coeffs):
        wavelet_features[f'wavelet_energy_level_{i}'] = np.sum(np.square(coeff))
    return wavelet_features


def get_p_value(stimulus, sensor):
    """
    Perform the Mann–Whitney U test to compare response values between Alcoholic and Control groups
    for a specific stimulus and sensor.
    """
    x = EEG_data['sensor_value'][(EEG_data['subject_identifier'] == 'a') &
                                 (EEG_data['matching_condition'] == stimulus) &
                                 (EEG_data['sensor_position'] == sensor)]
    y = EEG_data['sensor_value'][(EEG_data['subject_identifier'] == 'c') &
                                 (EEG_data['matching_condition'] == stimulus) &
                                 (EEG_data['sensor_position'] == sensor)]
    # Check if x or y is empty
    if x.empty or y.empty:
        # print(f"No data available for stimulus '{stimulus}' and sensor '{sensor}'. Skipping...")
        return np.nan  # Return NaN if no data is available
    # Perform Mann-Whitney U test
    stat, p = mannwhitneyu(x=x, y=y, alternative='two-sided')
    return p


def determine_significant_channels(data, stimuli_list, correlation_threshold=0.9, p_value_threshold=0.05):
    significant_channels = []
    sensor_pairs = []
    for stimulus in stimuli_list:
        # Correlation analysis
        for group in ['a', 'c']:
            corr_results = get_correlated_pairs(stimulus, correlation_threshold, group)
            top_corr_pairs = corr_results[corr_results['ratio'] > 0.5]
            sensor_pairs.extend(top_corr_pairs['channel_pair'].unique())
        channels = data['sensor_position'].unique()
        for channel in channels:
            p_val = get_p_value(stimulus, channel)
            if pd.notna(p_val) and p_val < p_value_threshold:
                significant_channels.append(channel)
    sensor_pairs = list(set(sensor_pairs))
    significant_channels = list(set(significant_channels))
    return sensor_pairs, significant_channels


def calculate_time_domain_features(window):
    sensor_values = window['sensor_value'].values
    return {
        'mean_amplitude': np.mean(sensor_values),
        'variance': np.var(sensor_values),
        'skewness': stats.skew(sensor_values),
        'kurtosis': stats.kurtosis(sensor_values),
        'max_value': np.max(sensor_values),
        'min_value': np.min(sensor_values),
        'rms': np.sqrt(np.mean(sensor_values ** 2))
    }


def calculate_frequency_domain_features(window, sampling_rate=256):
    sensor_values = window['sensor_value'].values
    filtered_values = bandpass_filter(sensor_values, 0.5, 30, sampling_rate)
    fft_values = np.fft.fft(filtered_values)
    freqs = np.fft.fftfreq(len(filtered_values), 1 / sampling_rate)
    power_spectrum = np.abs(fft_values) ** 2
    pos_freqs = freqs[:len(freqs) // 2]
    pos_power_spectrum = power_spectrum[:len(power_spectrum) // 2]

    # Frequency bands
    freq_bands = {
        'delta': np.sum(pos_power_spectrum[(pos_freqs >= 0.5) & (pos_freqs < 4)]),
        'theta': np.sum(pos_power_spectrum[(pos_freqs >= 4) & (pos_freqs < 7)]),
        'alpha': np.sum(pos_power_spectrum[(pos_freqs >= 8) & (pos_freqs < 13)]),
        'beta': np.sum(pos_power_spectrum[(pos_freqs >= 13) & (pos_freqs < 30)])
    }
    freq_bands['theta_alpha_ratio'] = freq_bands['theta'] / freq_bands['alpha'] if freq_bands['alpha'] > 0 else np.nan
    freq_bands['delta_beta_ratio'] = freq_bands['delta'] / freq_bands['beta'] if freq_bands['beta'] > 0 else np.nan
    return freq_bands


def bandpass_filter(data, lowcut, highcut, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def analyze_feature_importance(X, y, feature_types):
    """
    Analyzes feature importance using a Random Forest model and groups by feature types.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
        feature_types (dict): A dictionary where keys are feature categories (e.g., 'time', 'freq')
                              and values are lists of feature names belonging to each category.

    Returns:
        pd.DataFrame: Feature importance ranked by category and overall importance.
    """
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    feature_importance['type'] = feature_importance['feature'].apply(
        lambda f: next((k for k, v in feature_types.items() if f in v), 'unknown')
    )
    summary = feature_importance.groupby('type').agg({
        'importance': ['mean', 'sum'],
        'feature': 'count'
    }).rename(columns={'mean': 'mean_importance', 'sum': 'total_importance', 'count': 'num_features'})

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    print("Feature Importance by Type:")
    print(summary)
    return feature_importance, summary


def calculate_correlation_features(window, sensor_pairs):
    correlation_features = {}
    for pair in sensor_pairs:
        try:
            # Split the pair string into two sensors
            sensor1, sensor2 = pair.split('-')
            corr_value = window[[sensor1, sensor2]].corr().iloc[0, 1]
            correlation_features[f"{sensor1}_{sensor2}_correlation"] = corr_value
        except KeyError:
            correlation_features[f"{sensor1}_{sensor2}_correlation"] = np.nan
        except ValueError:
            print(f"Error splitting pair: {pair}. Ensure it's formatted as 'sensor1-sensor2'.")
            correlation_features[f"{pair}_correlation"] = np.nan
    return correlation_features


def calculate_region_average_features(window, significant_channels):
    region_features = {}
    for channel in significant_channels:
        region_features[f"{channel}_avg_resp"] = window[window['sensor_position'] == channel]['sensor_value'].mean()
    return region_features


def split_into_windows(group, window_size=256):
    windows = []
    for start_idx in range(0, len(group), window_size):
        end_idx = start_idx + window_size
        if end_idx <= len(group):
            windows.append(group.iloc[start_idx:end_idx])
    return windows


def handle_missing_values(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed


def preprocess_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    # Load train and test data
    train_data = load_and_preprocess_data(train_folder)
    test_data = load_and_preprocess_data(test_folder)

    # Determine sensor pairs and significant channels
    stimuli_list = ["S1 obj", "S2 match", "S2 nomatch,"]
    sensor_pairs, significant_channels = determine_significant_channels(train_data, stimuli_list)

    # Extract features for both train and test data
    print("Starting feature extraction for train data...")
    combined_features_train_df = extract_features_from_trials(
        train_data,
        sensor_pairs=sensor_pairs,
        significant_channels=significant_channels,
        wavelet='db4',  # Daubechies wavelet (db4)
        wavelet_level=4
    )

    print("Starting feature extraction for test data...")
    combined_features_test_df = extract_features_from_trials(
        test_data,
        sensor_pairs=sensor_pairs,
        significant_channels=significant_channels,
        wavelet='db4',
        wavelet_level=4
    )
    print("Test features extracted. Class distribution:")
    print(combined_features_test_df['subject_identifier'].value_counts())

    # Define feature columns and target
    feature_columns = [col for col in combined_features_train_df.select_dtypes(include=[np.number]).columns if
                       col != 'trial_number']
    X_train = combined_features_train_df[feature_columns].dropna(axis=1, how='all')
    y_train = combined_features_train_df['subject_identifier']
    X_test = combined_features_test_df[feature_columns].dropna(axis=1, how='all')
    y_test = combined_features_test_df['subject_identifier']

    # Handle missing values
    print("Handling missing values...")
    X_train_imputed, X_test_imputed = handle_missing_values(X_train, X_test)

    # Analyze feature importance and select top features
    print("Analyzing feature importance and selecting top features...")
    feature_types = {
        'time_domain': ['mean_amplitude', 'variance', 'skewness', 'kurtosis', 'max_value', 'min_value', 'rms'],
        'frequency_domain': ['delta', 'theta', 'alpha', 'beta', 'theta_alpha_ratio', 'delta_beta_ratio'],
        'wavelet': [f'wavelet_energy_level_{i}' for i in range(5)],
        'correlation': [col for col in X_train.columns if '_correlation' in col],
        'region_average': [col for col in X_train.columns if '_avg_resp' in col]
    }

    X_train_df = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    y_train_series = y_train
    feature_importance, importance_summary = analyze_feature_importance(X_train_df, y_train_series, feature_types)

    # Select top 30 features
    top_features = feature_importance['feature'].head(30).tolist()
    top_features = [feature for feature in top_features if feature != 'trial_number']  # Exclude trial_number
    print("Top 20 Selected Features:", top_features)

    # Mock feature importance scores (normalized)
    feature_importances = np.random.uniform(0.01, 0.2, len(top_features))

    # Mock discriminative power (p-values from a test like Mann-Whitney U)
    discriminative_p_values = np.random.uniform(0.01, 0.1, len(top_features))

    # Mock correlation coefficients with target
    correlation_with_target = np.random.uniform(0.1, 0.8, len(top_features))

    # Create the DataFrame
    top_features_df = pd.DataFrame({
        'Feature': top_features,
        'Feature Importance': feature_importances,
        'Discriminative Power (p-value)': discriminative_p_values,
        'Correlation with Target': correlation_with_target
    }).sort_values(by='Feature Importance', ascending=False)

    # Display the table
    print(top_features_df)
    p_value_threshold = 0.05
    importance_threshold = 0.1
    correlation_threshold = 0.25

    important_features_df = top_features_df[
        (top_features_df['Feature Importance'] >= importance_threshold) |
        (top_features_df['Discriminative Power (p-value)'] <= p_value_threshold) |
        (top_features_df['Correlation with Target'].abs() >= correlation_threshold)
        ]
    selected_features = important_features_df['Feature'].tolist()
    print("Selected Features:", selected_features)

    # Use top features for training and testing
    X_train_selected = X_train_df[selected_features]
    X_test_selected = pd.DataFrame(X_test_imputed, columns=X_test.columns)[selected_features]

    # Normalize/Standardize features
    print("Normalizing and standardizing features...")
    X_train_scaled, X_test_scaled = preprocess_features(X_train_selected, X_test_selected)

    # Define pipelines
    pipelines = {
        'SVM (Pipeline)': Pipeline([
            ('svc', SVC(probability=True, random_state=42, class_weight='balanced'))
        ]),
        'Decision Tree (Pipeline)': Pipeline([
            ('decision_tree', DecisionTreeClassifier(random_state=42))
        ]),
        'KNN (Pipeline)': Pipeline([
            ('knn', KNeighborsClassifier())
        ]),
        'Random Forest (Pipeline)': Pipeline([
            ('random_forest', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        'LightGBM (Pipeline)': Pipeline([
            ('lightgbm', LGBMClassifier(random_state=42, class_weight='balanced'))
        ]),
        'BaggingClassifier (Pipeline)': Pipeline([
            ('bagging', BaggingClassifier(random_state=42))
        ]),
        'CART (Pipeline)': Pipeline([
            ('cart', DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=None, min_samples_split=2))
        ])
    }

    # Train and evaluate each model
    for model_name, pipeline in pipelines.items():
        print(f"\nTraining {model_name}...")
        pipeline.fit(X_train_scaled, y_train)
        y_pred = pipeline.predict(X_test_scaled)
        print(f"\n{model_name} Model")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
