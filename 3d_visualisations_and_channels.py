import os
import random
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio  # Use pio.show for Plotly visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm import tqdm

###################################### README ########################################
# Add path to your local folder here. GitHub has a file size limit of 100 MB per file.
# Files can be downloaded here> https://www.kaggle.com/datasets/nnair25/Alcoholics/data
train_folder = "/Users/kristina/Dropbox/Mac/Desktop/Train copy"
######################################################################################

seed = 123
random.seed(seed)


def load_and_preprocess_data(folder_path):
    """
        Load and preprocess EEG data from a specified folder path.
        Steps:
        - Load CSV files, handling potential encoding issues.
        - Drop irrelevant columns and rename certain sensor positions.
        - Remove rows with undefined sensor positions.

        Args:
            folder_path (str): Path to the folder containing EEG data CSV files.

        Returns:
            pd.DataFrame: A cleaned and concatenated DataFrame containing EEG data.
        """
    filenames_list = os.listdir(folder_path)
    EEG_data_list = []
    for file_name in tqdm(filenames_list, desc="Loading Files"):
        file_path = os.path.join(folder_path, file_name)
        try:
            temp_df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            temp_df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip')
        EEG_data_list.append(temp_df)
    EEG_data = pd.concat(EEG_data_list, ignore_index=True)
    if 'Unnamed: 0' in EEG_data.columns:
        EEG_data = EEG_data.drop(['Unnamed: 0'], axis=1)

    EEG_data.loc[EEG_data['matching condition'] == 'S2 nomatch,', 'matching condition'] = 'S2 nomatch'
    EEG_data.loc[EEG_data['sensor position'] == 'AF1', 'sensor position'] = 'AF3'
    EEG_data.loc[EEG_data['sensor position'] == 'AF2', 'sensor position'] = 'AF4'
    EEG_data.loc[EEG_data['sensor position'] == 'PO1', 'sensor position'] = 'PO3'
    EEG_data.loc[EEG_data['sensor position'] == 'PO2', 'sensor position'] = 'PO4'
    EEG_data = EEG_data[(EEG_data['sensor position'] != 'X') &
                        (EEG_data['sensor position'] != 'Y') &
                        (EEG_data['sensor position'] != 'nd')]

    return EEG_data


EEG_data = load_and_preprocess_data(train_folder)
print("Data loaded and preprocessed. Sample data:")
print(EEG_data.head())


def sample_data(EEG_data, stimulus, random_id=random.randint(0, 7)):
    """
       Sample EEG data for a specific stimulus from both Alcoholic and Control groups.

       Args:
           EEG_data (pd.DataFrame): EEG data containing both groups and stimuli.
           stimulus (str): The stimulus condition (e.g., 'S1 obj', 'S2 match').
           random_id (int): Randomly selected subject identifier for each group.

       Returns:
           pd.DataFrame: Combined DataFrame of EEG data for the sampled subjects from both groups.
       """
    alcoholic_id = EEG_data['name'][(EEG_data['subject identifier'] == 'a') &
                                    (EEG_data['matching condition'] == stimulus)].unique()[random_id]
    control_id = EEG_data['name'][(EEG_data['subject identifier'] == 'c') &
                                  (EEG_data['matching condition'] == stimulus)].unique()[random_id]

    alcoholic_trial_number = EEG_data['trial number'][
        (EEG_data['name'] == alcoholic_id) & (EEG_data['matching condition'] == stimulus)].min()
    control_trial_number = EEG_data['trial number'][
        (EEG_data['name'] == control_id) & (EEG_data['matching condition'] == stimulus)].min()

    alcoholic_df = EEG_data[(EEG_data['name'] == alcoholic_id) & (EEG_data['trial number'] == alcoholic_trial_number)]
    control_df = EEG_data[(EEG_data['name'] == control_id) & (EEG_data['trial number'] == control_trial_number)]

    return pd.concat([alcoholic_df, control_df], ignore_index=True)


def get_correlated_pairs(stimulus, threshold, group):
    """
    Identify pairs of EEG sensor positions with high correlation for a specified stimulus and group.

    Args:
        stimulus (str): Stimulus condition (e.g., 'S1 obj', 'S2 match').
        threshold (float): Correlation threshold for identifying highly correlated pairs.
        group (str): Subject group ('a' for Alcoholic, 'c' for Control).

    Returns:
        pd.DataFrame: DataFrame of channel pairs, correlation counts, and normalized ratios.
    """
    corr_pairs_dict = {}
    trial_numbers_list = EEG_data['trial number'][
        (EEG_data['subject identifier'] == group) &
        (EEG_data['matching condition'] == stimulus)
        ].unique()

    sample_correlation_df = pd.pivot_table(
        EEG_data[(EEG_data['subject identifier'] == group) & (EEG_data['trial number'] == trial_numbers_list[0])],
        values='sensor value', index='sample num', columns='sensor position'
    ).corr()
    list_of_pairs = [f"{col}-{sample_correlation_df.index[i]}"
                     for j, col in enumerate(sample_correlation_df.columns)
                     for i in range(j + 1, len(sample_correlation_df))]

    corr_pairs_dict = {pair: 0 for pair in list_of_pairs}

    for trial_number in trial_numbers_list:
        correlation_df = pd.pivot_table(
            EEG_data[(EEG_data['subject identifier'] == group) & (EEG_data['trial number'] == trial_number)],
            values='sensor value', index='sample num', columns='sensor position'
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
    corr_count['ratio'] = corr_count['count'] / len(trial_numbers_list)  # Normalize by total trials
    return corr_count


def plot_3d_surface_and_heatmap(EEG_data, stimulus, group):
    """
        Plot 3D surface and heatmap of EEG sensor values for a specific group and stimulus.

        Args:
            EEG_data (pd.DataFrame): EEG data containing both groups and stimuli.
            stimulus (str): Stimulus condition.
            group (str): Subject group ('c' for Control, 'a' for Alcoholic).
        """
    if group == 'c':
        group_name = 'Control'
    else:
        group_name = 'Alcoholic'

    df = EEG_data[(EEG_data['subject identifier'] == group) &
                  (EEG_data['matching condition'] == stimulus)]
    temp_df = pd.pivot_table(df, index='channel', columns='sample num', values='sensor value').values.tolist()
    data = [go.Surface(z=temp_df, colorscale='Bluered')]

    layout = go.Layout(
        title=f'3D Surface and Heatmap for {stimulus} Stimulus - {group_name} Group',
        width=800,
        height=900,
        scene=dict(
            xaxis=dict(title='Time (sample num)', backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(title='Channel', backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(title='Sensor Value', backgroundcolor='rgb(230, 230,230)'),
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=[dict(args=["type", "surface"], label="3D Surface", method="restyle"),
                     dict(args=["type", "heatmap"], label="Heatmap", method="restyle")]
        )]
    )
    fig = go.Figure(data=data, layout=layout)
    pio.show(fig)


def get_correlated_pairs_sample(threshold, correlation_df, group, list_of_pairs):
    corr_pairs_dict = {pair: 0 for pair in list_of_pairs}

    for j, column in enumerate(correlation_df.columns):
        for i in range(j + 1, len(correlation_df)):
            pair_name = f"{column}-{correlation_df.index[i]}"
            if (correlation_df.iloc[i, j] >= threshold) and (pair_name in corr_pairs_dict):
                corr_pairs_dict[pair_name] += 1

    corr_count = pd.DataFrame.from_dict(corr_pairs_dict, orient='index', columns=['count']).reset_index()
    corr_count = corr_count.rename(columns={'index': 'channel_pair'})

    high_corr_pairs = corr_count['channel_pair'][corr_count['count'] > 0].tolist()
    print(f"Channel pairs with correlation >= {threshold} in {group} group:")
    print(high_corr_pairs)


def generate_list_of_pairs(correlation_df):
    list_of_pairs = [
        f"{col}-{correlation_df.index[i]}"
        for j, col in enumerate(correlation_df.columns)
        for i in range(j + 1, len(correlation_df))
    ]
    return list_of_pairs


def plot_sensors_correlation(df, threshold_value):
    correlations_alcoholic = pd.pivot_table(df[df['subject identifier'] == 'a'],
                                            values='sensor value', index='sample num', columns='sensor position').corr()
    correlations_control = pd.pivot_table(df[df['subject identifier'] == 'c'],
                                          values='sensor value', index='sample num', columns='sensor position').corr()

    list_of_pairs_alcoholic = [f"{col}-{correlations_alcoholic.index[i]}"
                               for j, col in enumerate(correlations_alcoholic.columns)
                               for i in range(j + 1, len(correlations_alcoholic))]

    list_of_pairs_control = [f"{col}-{correlations_control.index[i]}"
                             for j, col in enumerate(correlations_control.columns)
                             for i in range(j + 1, len(correlations_control))]

    fig = plt.figure(figsize=(17, 10))

    ax = fig.add_subplot(121)
    ax.set_title('Alcoholic group', fontsize=14)
    mask = np.zeros_like(correlations_alcoholic, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(correlations_alcoholic, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Plot for the Control group
    ax = fig.add_subplot(122)
    ax.set_title('Control group', fontsize=14)
    mask = np.zeros_like(correlations_control, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(correlations_control, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.suptitle('Correlation between Sensor Positions for ' + df['matching condition'].unique()[0] + ' stimulus',
                 fontsize=16)
    plt.show()

    # Display high-correlation pairs
    print(f"\nFinding correlated pairs for threshold {threshold_value} in Alcoholic group:")
    get_correlated_pairs_sample(threshold=threshold_value, correlation_df=correlations_alcoholic, group='Alcoholic',
                                list_of_pairs=list_of_pairs_alcoholic)
    print('\n')

    print(f"Finding correlated pairs for threshold {threshold_value} in Control group:")
    get_correlated_pairs_sample(threshold=threshold_value, correlation_df=correlations_control, group='Control',
                                list_of_pairs=list_of_pairs_control)


def compare_corr_pairs(stimulus):
    control_df = corr_pairs_df[(corr_pairs_df['stimulus'] == stimulus) & (corr_pairs_df['group'] == 'c')]
    alcoholic_df = corr_pairs_df[(corr_pairs_df['stimulus'] == stimulus) & (corr_pairs_df['group'] == 'a')]
    merged_df = pd.DataFrame(
        {'channel_pair': pd.concat([control_df['channel_pair'], alcoholic_df['channel_pair']]).unique()})
    merged_df = merged_df.merge(control_df[['channel_pair', 'ratio']], on='channel_pair', how='left').rename(
        columns={'ratio': 'ratio_control'})
    merged_df = merged_df.merge(alcoholic_df[['channel_pair', 'ratio']], on='channel_pair', how='left').rename(
        columns={'ratio': 'ratio_alcoholic'})
    merged_df = merged_df.fillna(0)
    merged_df['max_ratio'] = merged_df[['ratio_control', 'ratio_alcoholic']].max(axis=1)
    top_25_pairs = merged_df.sort_values(by='max_ratio', ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    index = range(len(top_25_pairs))

    ax.bar(index, top_25_pairs['ratio_control'], bar_width, label='Control Group', color='blue')
    ax.bar([i + bar_width for i in index], top_25_pairs['ratio_alcoholic'], bar_width, label='Alcoholic Group',
           color='orange')

    ax.set_title(f'Top 25 Correlated Pairs for Stimulus: {stimulus}')
    ax.set_xlabel('Channel Pairs')
    ax.set_ylabel('Ratio')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(top_25_pairs['channel_pair'], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()


def get_p_value(stimulus, sensor):
    """
    Perform the Mann–Whitney U test to compare response values between Alcoholic and Control groups for a specific stimulus and sensor.

    Args:
        stimulus (str): Stimulus condition.
        sensor (str): Sensor position.

    Returns:
        float: The p-value from the Mann–Whitney U test.
    """
    x = EEG_data['sensor value'][(EEG_data['subject identifier'] == 'a') &
                                 (EEG_data['matching condition'] == stimulus) &
                                 (EEG_data['sensor position'] == sensor)]
    y = EEG_data['sensor value'][(EEG_data['subject identifier'] == 'c') &
                                 (EEG_data['matching condition'] == stimulus) &
                                 (EEG_data['sensor position'] == sensor)]

    stat, p = mannwhitneyu(x=x, y=y, alternative='two-sided')
    return p

stimuli_list = ["S1 obj", "S2 match", "S2 nomatch"]
groups = ["a", "c"]  # 'a' for Alcoholic, 'c' for Control
correlation_threshold = 0.97

correlation_results = {}

corr_pairs_df = pd.DataFrame({})
size_df = EEG_data.groupby(['subject identifier', 'matching condition'])[['trial number']].nunique().reset_index(
    drop=False).rename(columns={'trial number': 'trials_count'})


for stimulus in stimuli_list:
    corr_pairs_df = corr_pairs_df._append(get_correlated_pairs(stimulus=stimulus, threshold=.9, group='c'))
    corr_pairs_df = corr_pairs_df._append(get_correlated_pairs(stimulus=stimulus, threshold=.9, group='a'))
corr_pairs_df = corr_pairs_df.merge(size_df, left_on=['group', 'stimulus'], right_on=['subject identifier', 'matching condition'], how='left')
compare_corr_pairs(stimulus='S1 obj')
compare_corr_pairs(stimulus='S2 match')
compare_corr_pairs(stimulus='S2 nomatch')

for stimulus in stimuli_list:
    sampled_data = sample_data(EEG_data, stimulus=stimulus)

    for group in groups:
        print(f"Plotting for Stimulus: {stimulus} | Group: {'Alcoholic' if group == 'a' else 'Control'}")
        plot_3d_surface_and_heatmap(sampled_data, stimulus=stimulus, group=group)

    print(f"\nPlotting correlation heatmaps for {stimulus} stimulus:")
    plot_sensors_correlation(sampled_data, correlation_threshold)


stimulus_list = EEG_data['matching condition'].unique().tolist()
channels_list = EEG_data['sensor position'].unique().tolist()

agg_df = EEG_data.groupby(['subject identifier', 'matching condition', 'sensor position'], as_index=False)[
    ['sensor value']].mean()

stat_test_results_list = []

for sensor in tqdm(channels_list, desc="Running Mann-Whitney U tests"):
    for stimulus in stimulus_list:
        p_value = get_p_value(stimulus=stimulus, sensor=sensor)
        stat_test_results_list.append({'stimulus': stimulus, 'sensor': sensor, 'p_value': p_value})

stat_test_results = pd.DataFrame(stat_test_results_list)

stat_test_results['reject_null'] = stat_test_results['p_value'] <= 0.05

significance_ratio = stat_test_results.groupby(['stimulus'])['reject_null'].mean()
print("Ratio of significant differences across all channels for each stimulus:")
print(significance_ratio)

pivot_data = stat_test_results.pivot(index='sensor', columns='stimulus', values='reject_null')

fig, ax = plt.subplots(figsize=(12, 6))

for stimulus in stimulus_list:
    ax.bar(pivot_data.index, pivot_data[stimulus], label=stimulus)

ax.set_title('Amount of Significant Differences for each Channel')
ax.set_xlabel('Sensor Position')
ax.set_ylabel('Significance Ratio (Proportion)')
ax.set_xticks(range(len(pivot_data.index)))
ax.set_xticklabels(pivot_data.index, rotation=45, ha="right")

ax.legend(title='Stimulus')
plt.tight_layout()
plt.show()
