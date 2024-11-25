import eeg_features as ef
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# Enter file paths for train and test EEG data folders
train_folder = r""
test_folder = r""

"""
Prepares sequences of EEG data for training/testing by splitting data into fixed-size windows.
Args:
    trials (pd.DataFrame): DataFrame containing EEG trial data.
    window_size (int): Number of samples per window.
Returns:
    np.array: Array of EEG signal sequences.
    np.array: Corresponding labels for each sequence.
"""
def prepare_sequences(trials, window_size=256):
    sequences = []
    labels = []
    grouped = trials.groupby(['trial_number', 'matching_condition'])
    for name, group in grouped:
        sensor_positions = group['sensor_position'].unique()
        for sensor in sensor_positions:
            sensor_data = group[group['sensor_position'] == sensor]
            windows = ef.split_into_windows(sensor_data, window_size)
            for window in windows:
                if len(window) == window_size:
                    sequences.append(window['sensor_value'].values)
                    labels.append(window['subject_identifier'].iloc[0])
    return np.array(sequences), np.array(labels)

def main(): 
    # Load train and test EEG trials
    train_trials = ef.load_all_trials_from_folder(train_folder)
    test_trials = ef.load_all_trials_from_folder(test_folder)

    # Prepare sequences and labels for training and testing
    X_train_seq, y_train_seq = prepare_sequences(train_trials)
    X_test_seq, y_test_seq = prepare_sequences(test_trials)

    # Encode labels into numeric format
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_seq)
    y_test_encoded = label_encoder.transform(y_test_seq)


    # Standardize the features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, X_train_seq.shape[-1])).reshape(X_train_seq.shape)
    X_test_scaled = scaler.transform(X_test_seq.reshape(-1, X_test_seq.shape[-1])).reshape(X_test_seq.shape)
    
    # Reshape data for input into RNN models
    X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    # Build the RNN model using convolutional, GRU, and dense layers
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(32))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with appropriate loss function and metrics
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Compute class weights to address class imbalance
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weight_dict = dict(enumerate(class_weights))

    # Train the model with training data and validate using test data
    history = model.fit(X_train_rnn, y_train_encoded, epochs=50, batch_size=32, validation_data=(X_test_rnn, y_test_encoded), class_weight=class_weight_dict)

    # Predict on test data
    y_pred = model.predict(X_test_rnn)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Generate and print classification report
    print(classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_))

    # Compute and print confusion matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_classes)
    print(conf_matrix)
    
    
    # Plot training and validation loss over epochs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy over epochs
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
    
if __name__ == '__main__':
    main()