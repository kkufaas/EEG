import os
import pandas as pd
from scipy.stats import mannwhitneyu
from prettytable import PrettyTable

# Sensor location mappings
sensor_to_location = {
    'FP1': 'Frontal', 'FP2': 'Frontal', 'F7': 'Frontal', 'F8': 'Frontal',
    'AF1': 'Frontal', 'AF2': 'Frontal', 'FZ': 'Frontal', 'F4': 'Frontal',
    'F3': 'Frontal', 'FC6': 'Frontal-Central', 'FC5': 'Frontal-Central',
    'FC2': 'Frontal-Central', 'FC1': 'Frontal-Central', 'T8': 'Temporal',
    'T7': 'Temporal', 'CZ': 'Central', 'C3': 'Central', 'C4': 'Central',
    'CP5': 'Central-Parietal', 'CP6': 'Central-Parietal', 'CP1': 'Central-Parietal',
    'CP2': 'Central-Parietal', 'P3': 'Parietal', 'P4': 'Parietal',
    'PZ': 'Parietal', 'P8': 'Parietal', 'P7': 'Parietal', 'PO2': 'Parietal-Occipital',
    'PO1': 'Parietal-Occipital', 'O2': 'Occipital', 'O1': 'Occipital', 'X': 'Unknown',
    'AF7': 'Frontal', 'AF8': 'Frontal', 'F5': 'Frontal', 'F6': 'Frontal',
    'FT7': 'Frontal-Temporal', 'FT8': 'Frontal-Temporal', 'FPZ': 'Frontal',
    'FC4': 'Frontal-Central', 'FC3': 'Frontal-Central', 'C6': 'Central',
    'C5': 'Central', 'F2': 'Frontal', 'F1': 'Frontal', 'TP8': 'Temporal-Parietal',
    'TP7': 'Temporal-Parietal', 'AFZ': 'Frontal', 'CP3': 'Central-Parietal',
    'CP4': 'Central-Parietal', 'P5': 'Parietal', 'P6': 'Parietal', 'C1': 'Central',
    'C2': 'Central', 'PO7': 'Parietal-Occipital', 'PO8': 'Parietal-Occipital',
    'FCZ': 'Frontal-Central', 'POZ': 'Parietal-Occipital', 'OZ': 'Occipital',
    'P2': 'Parietal', 'P1': 'Parietal', 'CPZ': 'Central-Parietal', 'nd': 'Undefined',
    'Y': 'Unknown'
}


def map_sensors_to_locations(unique_sensors, mapping):
    """
    Maps EEG sensors to their corresponding anatomical locations.

    Args:
        unique_sensors (list): List of unique sensor names.
        mapping (dict): Dictionary mapping sensor names to locations.

    Returns:
        dict: Dictionary with sensor names as keys and their locations as values.
    """
    sensor_location_map = {sensor: mapping.get(sensor, 'Undefined') for sensor in unique_sensors}
    return sensor_location_map


def analyze_sensor_correlations_and_tests(EEG_data, stimuli_list, correlation_threshold=0.97, p_value_threshold=0.05):
    """
    Analyze sensor correlations and perform Mann-Whitney U tests to identify significant differences.

    Args:
        EEG_data (pd.DataFrame): EEG data with sensor values.
        stimuli_list (list): List of stimuli to analyze.
        correlation_threshold (float): Correlation threshold to identify significant pairs.
        p_value_threshold (float): Threshold for Mann-Whitney U test significance.

    Returns:
        None: Displays formatted tables and summaries.
    """
    significant_sensor_pairs = {stimulus: {"a": [], "c": []} for stimulus in stimuli_list}
    mann_whitney_summary = {}
    total_pairs_summary = {"a": {}, "c": {}}
    location_distribution_summary = {stimulus: {"a": {}, "c": {}} for stimulus in stimuli_list}

    for stimulus in stimuli_list:
        # Get data for the specific stimulus
        for group in ["a", "c"]:
            group_data = EEG_data[(EEG_data["subject_identifier"] == group) &
                                  (EEG_data["matching_condition"] == stimulus)]
            correlation_matrix = group_data.pivot_table(index="sample_num", columns="sensor_position",
                                                        values="sensor_value").corr()

            # Collect significant pairs and their locations
            location_counts = {}
            for sensor1 in correlation_matrix.columns:
                for sensor2 in correlation_matrix.columns:
                    if sensor1 != sensor2 and correlation_matrix.at[sensor1, sensor2] >= correlation_threshold:
                        location1 = sensor_to_location.get(sensor1, "Unknown")
                        location2 = sensor_to_location.get(sensor2, "Unknown")
                        significant_sensor_pairs[stimulus][group].append(
                            f"{sensor1} ({location1}) - {sensor2} ({location2})"
                        )
                        # Update location counts
                        location_counts[location1] = location_counts.get(location1, 0) + 1
                        location_counts[location2] = location_counts.get(location2, 0) + 1

            # Calculate location distribution as percentages
            total_locations = sum(location_counts.values())
            if total_locations > 0:
                location_distribution_summary[stimulus][group] = {loc: (count / total_locations) * 100
                                                                  for loc, count in location_counts.items()}

        # Perform Mann-Whitney U tests
        p_values = []
        for sensor in EEG_data["sensor_position"].unique():
            group_a = EEG_data[(EEG_data["subject_identifier"] == "a") &
                               (EEG_data["matching_condition"] == stimulus) &
                               (EEG_data["sensor_position"] == sensor)]["sensor_value"]

            group_c = EEG_data[(EEG_data["subject_identifier"] == "c") &
                               (EEG_data["matching_condition"] == stimulus) &
                               (EEG_data["sensor_position"] == sensor)]["sensor_value"]

            if not group_a.empty and not group_c.empty:
                stat, p_value = mannwhitneyu(group_a, group_c, alternative="two-sided")
                if p_value <= p_value_threshold:
                    p_values.append(p_value)

        mann_whitney_summary[stimulus] = (len(p_values) / len(EEG_data["sensor_position"].unique())) * 100
        total_pairs_summary["a"][stimulus] = len(significant_sensor_pairs[stimulus]["a"])
        total_pairs_summary["c"][stimulus] = len(significant_sensor_pairs[stimulus]["c"])

    # Display results
    for stimulus in stimuli_list:
        print(f"\nStimulus: {stimulus}")
        print(f"Percentage of Significant Sensors (Mann-Whitney Test): {mann_whitney_summary[stimulus]:.2f}%")

        print("Location Distribution of Significant Sensor Pairs (Alcoholic Group):")
        table_a_loc = PrettyTable()
        table_a_loc.field_names = ["Location", "Percentage"]
        for loc, perc in location_distribution_summary[stimulus]["a"].items():
            table_a_loc.add_row([loc, f"{perc:.2f}%"])
        print(table_a_loc)

        print("Location Distribution of Significant Sensor Pairs (Control Group):")
        table_c_loc = PrettyTable()
        table_c_loc.field_names = ["Location", "Percentage"]
        for loc, perc in location_distribution_summary[stimulus]["c"].items():
            table_c_loc.add_row([loc, f"{perc:.2f}%"])
        print(table_c_loc)

    # Summary
    print("\nSummary:")
    for stimulus in stimuli_list:
        print(f"Stimulus: {stimulus}")
        print(f"Total significant pairs for Alcoholic Group: {total_pairs_summary['a'][stimulus]}")
        print(f"Total significant pairs for Control Group: {total_pairs_summary['c'][stimulus]}")

    total_pairs_a = sum(total_pairs_summary["a"].values())
    total_pairs_c = sum(total_pairs_summary["c"].values())
    print(f"\nOverall Total significant pairs for Alcoholic Group: {total_pairs_a}")
    print(f"Overall Total significant pairs for Control Group: {total_pairs_c}")


def load_and_preprocess_data(folder_path):
    columns = ['trial_number', 'sensor_position', 'sample_num', 'sensor_value',
               'subject_identifier', 'matching_condition', 'channel', 'name', 'time']
    all_trials = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            trial_data = pd.read_csv(file_path, sep=',', header=0, names=columns)
            all_trials.append(trial_data)
    return pd.concat(all_trials, ignore_index=True)


def extract_unique_sensors(EEG_data):
    """
    Extract all unique sensor names from EEG data.

    Args:
        EEG_data (pd.DataFrame): The EEG dataset.

    Returns:
        list: A list of unique sensor names.
    """
    unique_sensors = EEG_data['sensor_position'].unique()
    print("Unique Sensor Names:")
    print(unique_sensors)
    return unique_sensors


# Data source folder
train_folder = "/Users/kristina/Dropbox/Mac/Desktop/Train copy"
EEG_data = load_and_preprocess_data(train_folder)
stimuli_list = ["S1 obj", "S2 match", "S2 nomatch,"]
# extract_unique_sensors(EEG_data)
analyze_sensor_correlations_and_tests(EEG_data, stimuli_list)
