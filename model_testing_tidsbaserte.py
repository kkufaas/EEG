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
import matplotlib.pyplot as plt

###################################### README ########################################
# Add path to your local folder here. GitHub has a file size limit of 100 MB per file.
# Files can be downloaded here> https://www.kaggle.com/datasets/nnair25/Alcoholics/data
train_folder = "/Users/kristina/Dropbox/Mac/Desktop/Train copy"
######################################################################################

columns_raw_data = [
    'trial_number', 'sensor_position', 'sample_num', 'sensor_value',
    'subject_identifier', 'matching_condition', 'channel', 'name', 'time'
]

# DEBUG: sets pandas options to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def load_all_trials_from_folder(folder_path):
    """ Load all CSV files from the specified folder and return a concatenated DataFrame."""
    all_trials = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            trial_data = pd.read_csv(file_path, sep=',', header=0, names=columns_raw_data)
            all_trials.append(trial_data)
    return pd.concat(all_trials, ignore_index=True)


def split_into_windows(group, window_size=256):
    """ Split a group of sensor data into windows of a fixed size (256 samples per window). """
    n_samples = len(group)
    windows = []
    for start_idx in range(0, n_samples, window_size):
        end_idx = start_idx + window_size
        if end_idx <= n_samples:
            windows.append(group.iloc[start_idx:end_idx])
    return windows


def calculate_time_domain_features(window):
    """ Calculate time-domain statistical features from an EEG signal window. """
    sensor_values = window['sensor_value'].values

    features = {
        # Mean Amplitude
        'mean_amplitude': np.mean(sensor_values),

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
    """
    feature_list = []
    grouped = trials.groupby(['trial_number', 'sensor_position', 'matching_condition'])
    for name, group in grouped:
        subject_identifier = group['subject_identifier'].iloc[0]
        patient_id = group['name'].iloc[0]
        windows = split_into_windows(group, window_size)

        for window in windows:
            features = calculate_time_domain_features(window)

            features['sensor_position'] = name[1]
            features['subject_identifier'] = subject_identifier
            features['matching_condition'] = group['matching_condition'].iloc[0]
            features['name'] = patient_id

            feature_list.append(features)
    return pd.DataFrame(feature_list)


if __name__ == "__main__":
    # Load the data
    train_data = load_all_trials_from_folder(train_folder)

    # Extract time-domain features
    extracted_time_features_train_df = extract_features_from_trials(train_data)

    # Use relevant columns for classification
    feature_columns = ['mean_amplitude', 'variance', 'skewness', 'kurtosis', 'rms', 'min_value', 'max_value']
    X = extracted_time_features_train_df[feature_columns]
    y = extracted_time_features_train_df['subject_identifier']  # 'a' for alcoholic, 'c' for control

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Unique classes in y_train:", np.unique(y_train))
    print("Class distribution in y_train:\n", y_train.value_counts())
    print("Class distribution in y_test:\n", y_test.value_counts())

    # Define pipelines with imputation for all models
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

    # Train the models and evaluate
    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"\n{model_name} Model")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

