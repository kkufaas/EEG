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
def calculate_frequency_band_power(sensor_values, sampling_rate=256):
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

    # Compute ratios
    freq_bands['theta_alpha_ratio'] = (
        freq_bands['theta'] / freq_bands['alpha'] if freq_bands['alpha'] > 0 else np.nan
    )
    freq_bands['delta_beta_ratio'] = (
        freq_bands['delta'] / freq_bands['beta'] if freq_bands['beta'] > 0 else np.nan
    )
    return freq_bands


def plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus, group, freq_band, sampling_rate=256, window_size=256):
    """
    Plot 3D surface and heatmap of frequency band power for a specific group and stimulus over time.

    Args:
        EEG_data (pd.DataFrame): EEG data containing both groups and stimuli.
        stimulus (str): Stimulus condition.
        group (str): Subject group ('c' for Control, 'a' for Alcoholic).
        freq_band (str): Frequency band ('delta', 'theta', 'alpha', or 'beta').
        sampling_rate (int): EEG sampling rate (default: 256 Hz).
        window_size (int): Size of the sliding window for frequency power calculation (default: 256 samples).
    """
    group_name = 'Control' if group == 'c' else 'Alcoholic'

    # Filter data based on group and stimulus
    df = EEG_data[(EEG_data['subject_identifier'] == group) &
                  (EEG_data['matching_condition'] == stimulus)]

    # Get unique sensor positions
    sensors = df['sensor_position'].unique()

    # Determine the number of time windows
    sample_data = df[df['sensor_position'] == sensors[0]]['sensor_value'].values
    num_windows = (len(sample_data) - window_size) // (window_size // 2) + 1

    # Prepare an empty DataFrame to store frequency band power for each sensor over time
    freq_power_matrix = pd.DataFrame(index=sensors, columns=range(num_windows))

    for channel in sensors:
        channel_data = df[df['sensor_position'] == channel]['sensor_value'].values
        # Split the channel data into overlapping windows
        time_series_power = []
        for i in range(num_windows):
            start_idx = i * (window_size // 2)
            end_idx = start_idx + window_size
            window_data = channel_data[start_idx:end_idx]
            if len(window_data) < window_size:
                break  # Skip incomplete window
            freq_bands = calculate_frequency_band_power(window_data, sampling_rate)
            time_series_power.append(freq_bands[freq_band])

        # Add the time-series power to the matrix
        freq_power_matrix.loc[channel, :] = time_series_power

    # Generate 3D surface plot data
    x = np.arange(freq_power_matrix.shape[1])  # Time windows
    y = np.arange(freq_power_matrix.shape[0])  # Channels
    x_grid, y_grid = np.meshgrid(x, y)
    z = freq_power_matrix.values.astype(float)

    # Create the plot
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=x_grid,
                y=y_grid,
                colorscale='Viridis',
                colorbar=dict(title=f'{freq_band.capitalize()} Power')
            )
        ]
    )

    # Customize layout
    fig.update_layout(
        title=f'3D Surface Plot for {freq_band.capitalize()} Band ({stimulus} Stimulus - {group_name} Group)',
        scene=dict(
            xaxis=dict(title='Time (Window Index)'),
            yaxis=dict(title='Sensor Position'),
            zaxis=dict(title=f'{freq_band.capitalize()} Power')
        ),
        width=1400,
        height=1000
    )

    # Show the plot
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
    group_name = 'Control' if group == 'c' else 'Alcoholic'

    # Filter data based on group and stimulus
    df = EEG_data[(EEG_data['subject_identifier'] == group) &
                  (EEG_data['matching_condition'] == stimulus)]

    # Compute frequency band power for each channel
    channel_values = []
    for channel in df['sensor_position'].unique():
        channel_data = df[df['sensor_position'] == channel]['sensor_value'].values
        freq_bands = calculate_frequency_band_power(channel_data, sampling_rate)
        channel_values.append(freq_bands[freq_band])

    # Generate the 3D data
    pivot_data = pd.pivot_table(
        df,
        index='sensor_position',  # Rows: channels
        columns='sample_num',  # Columns: time/sample
        values='sensor_value'
    )

    # Replace raw sensor values with frequency band power
    for i, channel in enumerate(pivot_data.index):
        pivot_data.loc[channel, :] = channel_values[i]

    # Prepare 3D surface plot data
    x = np.arange(pivot_data.shape[1])  # Sample numbers
    y = np.arange(pivot_data.shape[0])  # Channels
    x_grid, y_grid = np.meshgrid(x, y)
    z = pivot_data.values

    # Create the plot
    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=x_grid,
                y=y_grid,
                colorscale='Viridis',
                colorbar=dict(title=f'{freq_band.capitalize()} Power')
            )
        ]
    )

    # Customize layout
    fig.update_layout(
        title=f'3D Surface Plot for {freq_band.capitalize()} Band ({stimulus} Stimulus - {group_name} Group)',
        scene=dict(
            xaxis=dict(title='Time (Sample Num)'),
            yaxis=dict(title='Channel'),
            zaxis=dict(title=f'{freq_band.capitalize()} Power')
        ),
        width=900,
        height=700
    )

    # Show the plot
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
train_folder = "/Users/kristina/Dropbox/Mac/Desktop/Train copy"  # Replace with your actual data path
EEG_data = load_and_preprocess_data(train_folder)

# Plot 3D visualization for specific frequency bands

# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S1 obj', group='a', freq_band='alpha')
# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S1 obj', group='c', freq_band='alpha')

# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 match', group='a', freq_band='alpha')
# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 match', group='c', freq_band='alpha')
#
# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='alpha')
# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='alpha')

# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S1 obj', group='a', freq_band='beta')
# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S1 obj', group='c', freq_band='beta')
#
# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 match', group='a', freq_band='beta')
# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 match', group='c', freq_band='beta')
#
# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='beta')
# plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='beta')


plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S1 obj', group='a', freq_band='delta')
plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S1 obj', group='c', freq_band='delta')

plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 match', group='a', freq_band='delta')
plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 match', group='c', freq_band='delta')

plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='delta')
plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='delta')


plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S1 obj', group='a', freq_band='theta')
plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S1 obj', group='c', freq_band='theta')

plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 match', group='a', freq_band='theta')
plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 match', group='c', freq_band='theta')

plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='theta')
plot_3d_surface_and_heatmap_freq_timevarying(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='theta')


# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='a', freq_band='alpha')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='c', freq_band='alpha')

# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='a', freq_band='alpha')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='c', freq_band='alpha')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='alpha')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='alpha')
#
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='a', freq_band='beta')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='c', freq_band='beta')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='a', freq_band='beta')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='c', freq_band='beta')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='beta')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='beta')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='a', freq_band='delta')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='c', freq_band='delta')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='a', freq_band='delta')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='c', freq_band='delta')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='delta')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='delta')
#
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='a', freq_band='theta')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='c', freq_band='theta')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='a', freq_band='theta')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='c', freq_band='theta')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='theta')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='theta')


# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='a', freq_band='theta_alpha_ratio')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='c', freq_band='theta_alpha_ratio')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='a', freq_band='theta_alpha_ratio')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='c', freq_band='theta_alpha_ratio')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='theta_alpha_ratio')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='theta_alpha_ratio')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='a', freq_band='delta_beta_ratio')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S1 obj', group='c', freq_band='delta_beta_ratio')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='a', freq_band='delta_beta_ratio')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 match', group='c', freq_band='delta_beta_ratio')
#
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='a', freq_band='delta_beta_ratio')
# plot_3d_surface_and_heatmap_freq(EEG_data, stimulus='S2 nomatch,', group='c', freq_band='delta_beta_ratio')
