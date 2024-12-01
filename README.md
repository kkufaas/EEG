
This is research based on EEG dataset (https://www.kaggle.com/datasets/nnair25/Alcoholics/data), where 
we try to investigate how alcohol affects brain responses to different stimuli, using EEG.

There are different files in this project, each responsible for own functionality. 
1. 3d_visualisation_sensor_position
Features:
3D visualizations: Interactive 3D surface and heatmap plots for sensor data.

Usage:
Place EEG CSV files in a folder and set train_folder path.
Run plotting functions to visualize group and stimulus data.

2. 3d_visualisation_frequency_bands
This script processes EEG data to compute and visualize frequency band power over time and across channels 
for different groups and stimuli.
Features:
Frequency analysis: Calculates power and ratios for EEG frequency bands.
Time-varying visualizations: Creates interactive 3D surface plots and heatmaps of frequency band power over time.

Usage:
Place EEG CSV files in a folder and set train_folder to the folder path.
Run the plotting functions for desired groups, stimuli, and frequency bands.

3. model_evaluation
This script processes EEG data for feature extraction, statistical analysis, 
and machine learning classification to differentiate between groups (control/alcoholic) based on EEG signals.
Features:
Feature extraction:
Time-domain, frequency-domain, wavelet, correlation, and regional average features.
Automatic selection of significant features based on statistical and model-based importance.
Statistical tests: Uses Mann–Whitney U test to identify significant channels and sensor pairs.
Machine learning pipelines:
SVM, Random Forest, LightGBM, Decision Tree, KNN, and Bagging Classifier.
Supports feature scaling and SMOTE for handling class imbalance.

Usage:
Place EEG CSV files in folders for training and testing data.
Set train_folder and test_folder to their respective paths.
Run the script to:
Extract features and identify significant ones.
Train and evaluate multiple classification models.

4. neural_network
This script processes EEG data for training and testing a neural network model combining convolutional (Conv1D) 
and recurrent (GRU) layers for classification tasks.
Features:
Sequence preparation: Splits EEG data into fixed-size windows for sequential analysis.
Data preprocessing:
Standardizes features for improved model performance.
Encodes labels into numeric format.
RNN model:
Conv1D, GRU, and Dense layers with dropout for regularization.
Handles class imbalance using computed class weights.
Evaluation:
Generates classification reports and confusion matrices.
Plots training and validation accuracy/loss over epochs.

Usage:
Specify train_folder and test_folder paths for EEG data.
Run the script to:
Preprocess data.
Train the RNN model.
Evaluate model performance.

5. sensor_position_and_significant_pairs_table
This script analyzes EEG data to identify significant sensor correlations and differences between groups (e.g., Control, Alcoholic) based on stimuli conditions.

Features:
Sensor mapping: Maps EEG sensors to anatomical locations.
Correlation analysis: Identifies highly correlated sensor pairs by group and stimulus.
Statistical testing: Uses Mann–Whitney U test to determine significant differences between groups.
Visualization: Displays location distributions and summaries in formatted tables.

Usage:
Place EEG CSV files in a folder and set train_folder to its path.
Run the script to:
Map sensors to locations.
Analyze correlations and significant sensor pairs by stimuli.
Generate statistical summaries.

6. time_domain_teatures_table
This script processes EEG data to calculate and summarize time-domain features for different groups and conditions.

Features:
Time-domain analysis:
Extracts features such as mean, variance, skewness, kurtosis, RMS, max, and min values.
Groups data by matching condition and subject identifier for targeted analysis.
Summary export: Saves calculated features as a CSV file for further analysis.

Usage:
Place EEG CSV files in a folder and set train_folder to its path.
Run the script to:
Calculate time-domain features.
Summarize features by group and condition.
Optionally export the summary as a CSV file.