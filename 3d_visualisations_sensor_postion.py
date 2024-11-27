import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.signal import butter, filtfilt


# Bandpass filter
def bandpass_filter(data, lowcut, highcut, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# Frequency band extraction
def calculate_frequency_domain_features(sensor_values, sampling_rate=256):
    filtered_values = bandpass_filter(sensor_values, 0.5, 30, sampling_rate)
    fft_values = np.fft.fft(filtered_values)
    freqs = np.fft.fftfreq(len(filtered_values), 1 / sampling_rate)
    power_spectrum = np.abs(fft_values) ** 2
    pos_freqs = freqs[:len(freqs) // 2]
    pos_power_spectrum = power_spectrum[:len(power_spectrum) // 2]

    freq_bands = {
        'delta': np.sum(pos_power_spectrum[(pos_freqs >= 0.5) & (pos_freqs < 4)]),
        'theta': np.sum(pos_power_spectrum[(pos_freqs >= 4) & (pos_freqs < 7)]),
        'alpha': np.sum(pos_power_spectrum[(pos_freqs >= 8) & (pos_freqs < 13)]),
        'beta': np.sum(pos_power_spectrum[(pos_freqs >= 13) & (pos_freqs < 30)])
    }
    return freq_bands


# 3D Surface and Heatmap Plot for sensor position
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

    # Filter the data for the specific group and stimulus
    df = EEG_data[(EEG_data['subject_identifier'] == group) &
                  (EEG_data['matching_condition'] == stimulus)]

    # Pivot the data to get a 2D matrix for plotting
    temp_df = pd.pivot_table(df, index='sensor_position', columns='sample_num', values='sensor_value')

    # Extract the values for the 3D plot
    z_values = temp_df.values
    y_labels = temp_df.index.tolist()  # Get the sensor positions as y-axis labels

    # Create the 3D surface plot
    data = [go.Surface(z=z_values, colorscale='Bluered')]

    # Configure the layout
    layout = go.Layout(
        title=f'3D Surface and Heatmap for {stimulus} Stimulus - {group_name} Group',
        width=800,
        height=900,
        scene=dict(
            xaxis=dict(title='Time (Sample Num)', backgroundcolor='rgb(230, 230, 230)'),
            yaxis=dict(
                title='Sensor Position',
                tickvals=list(range(len(y_labels))),  # Numeric positions for sensors
                ticktext=y_labels,  # Map these to sensor names
                backgroundcolor='rgb(230, 230, 230)'
            ),
            zaxis=dict(title='Sensor Value', backgroundcolor='rgb(230, 230, 230)')
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=[dict(args=["type", "surface"], label="3D Surface", method="restyle"),
                     dict(args=["type", "heatmap"], label="Heatmap", method="restyle")]
        )]
    )

    # Create the figure and show it
    fig = go.Figure(data=data, layout=layout)
    pio.show(fig)


# 3D Surface and Heatmap Plot for Frequency Bands
def plot_3d_surface_and_heatmap_freq(EEG_data, stimulus, group, freq_band, sampling_rate=256):
    """
    Plot 3D surface and heatmap of frequency band power for a specific group and stimulus.

    Args:
        EEG_data (pd.DataFrame): EEG data containing both groups and stimuli.
        stimulus (str): Stimulus condition.
        group (str): Subject group ('c' for Control, 'a' for Alcoholic).
        freq_band (str): Frequency band ('delta', 'theta', 'alpha', or 'beta').
        sampling_rate (int): EEG sampling rate (default: 256 Hz).
    """
    if group == 'c':
        group_name = 'Control'
    else:
        group_name = 'Alcoholic'

    df = EEG_data[(EEG_data['subject_identifier'] == group) &
                  (EEG_data['matching_condition'] == stimulus)]

    # Pivot table to structure data for channels vs. time
    freq_values = []
    for channel in df['sensor_position'].unique():
        channel_data = df[df['sensor_position'] == channel]['sensor_value'].values
        freq_bands = calculate_frequency_domain_features(channel_data, sampling_rate)
        freq_values.append(freq_bands[freq_band])

    temp_df = pd.pivot_table(df, index='sensor_position', columns='sample_num', values='sensor_value')
    temp_df.loc[:, :] = freq_values  # Replace with frequency band values

    # Create 3D surface plot
    data = [go.Surface(z=temp_df.values, colorscale='Viridis')]
    layout = go.Layout(
        title=f'3D Surface and Heatmap for {freq_band.capitalize()} Band ({stimulus} Stimulus - {group_name} Group)',
        width=800,
        height=900,
        scene=dict(
            xaxis=dict(title='Time (sample num)', backgroundcolor='rgb(230, 230, 230)'),
            yaxis=dict(title='Channel', backgroundcolor='rgb(230, 230, 230)'),
            zaxis=dict(title=f'{freq_band.capitalize()} Power', backgroundcolor='rgb(230, 230, 230)'),
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


# Load EEG data
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


# Specify your data folder
train_folder = "/Users/kristina/Dropbox/Mac/Desktop/Train copy"
EEG_data = load_and_preprocess_data(train_folder)

plot_3d_surface_and_heatmap(EEG_data, stimulus='S1 obj', group='a')
plot_3d_surface_and_heatmap(EEG_data, stimulus='S1 obj', group='c')

plot_3d_surface_and_heatmap(EEG_data, stimulus='S2 match', group='a')
plot_3d_surface_and_heatmap(EEG_data, stimulus='S2 match', group='c')

plot_3d_surface_and_heatmap(EEG_data, stimulus='S2 nomatch,', group='a')
plot_3d_surface_and_heatmap(EEG_data, stimulus='S2 nomatch,', group='c')
