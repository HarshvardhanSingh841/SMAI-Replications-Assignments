import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import shutil

# # Define the source and destination directories
# source_dir = '../../data/external/recordings'  # Path to your original recordings folder
# interim_dir = '../../data/interim/interim_recordings'  # Path to the new directory where files will be separated

# # Create the interim directory if it doesn't exist
# os.makedirs(interim_dir, exist_ok=True)

# # Create 10 subdirectories for digits 0-9
# for digit in range(10):
#     digit_folder = os.path.join(interim_dir, str(digit))
#     os.makedirs(digit_folder, exist_ok=True)

# # Iterate through all files in the source directory
# for filename in os.listdir(source_dir):
#     if filename.endswith('.wav'):  # Check if the file is a .wav file
#         # Extract the digit label (the part before the first underscore)
#         digit_label = filename.split('_')[0]
        
#         # Check if the digit is valid (it should be a number between 0 and 9)
#         if digit_label.isdigit() and int(digit_label) in range(10):
#             # Create the full path for the source file and the destination file
#             source_file = os.path.join(source_dir, filename)
#             destination_folder = os.path.join(interim_dir, digit_label)
#             destination_file = os.path.join(destination_folder, filename)
            
#             # Move the file to the appropriate folder
#             shutil.move(source_file, destination_file)

# print("Recordings successfully separated by digit labels!")



# interim_recordings_dir = '../../data/interim/interim_recordings'  # The folder containing the separated files
# interim_data_dir = '../../data/interim/interim_data'  # The folder where MFCCs will be saved

# # Create the interim_data directory if it doesn't exist
# os.makedirs(interim_data_dir, exist_ok=True)

# # Create 10 subdirectories for digits 0-9 inside the interim_data folder
# for digit in range(10):
#     digit_folder = os.path.join(interim_data_dir, str(digit))
#     os.makedirs(digit_folder, exist_ok=True)

# # Iterate over each folder in the interim_recordings directory (digits 0-9)
# for digit in range(10):
#     digit_folder = os.path.join(interim_recordings_dir, str(digit))  # Path to the digit folder
#     for filename in os.listdir(digit_folder):
#         if filename.endswith('.wav'):  # Only process .wav files
#             # Create the full path for the source file
#             source_file = os.path.join(digit_folder, filename)
            
#             # Load the audio file using librosa
#             y, sr = librosa.load(source_file, sr=None)
            
#             # Extract MFCC features (n_mfcc=13 is a common choice for speech)
#             mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,n_fft = 1024)
            
#             # Save the MFCC features as a .npy file in the corresponding digit folder
#             mfcc_filename = f"{filename.split('.')[0]}.npy"  # Use the same filename, but with .npy extension
#             mfcc_path = os.path.join(interim_data_dir, str(digit), mfcc_filename)
            
#             # Save the MFCC features to the .npy file
#             np.save(mfcc_path, mfcc)

# print("MFCC features successfully extracted and saved!")


# interim_data_dir = '../../data/interim/interim_data'  # Folder containing MFCCs

# # Set up the number of subplots (3x4 or 5x2)
# fig, axes = plt.subplots(5, 2, figsize=(15, 10))
# axes = axes.flatten()  # Flatten to easily access each axis

# # Loop through digits 0-9 to visualize their MFCCs
# for digit in range(10):
#     # Get a list of .npy files for the current digit
#     digit_folder = os.path.join(interim_data_dir, str(digit))
#     mfcc_files = [f for f in os.listdir(digit_folder) if f.endswith('.npy')]

#     # Select a recording (we can take the first one or randomly pick)
#     mfcc_path = os.path.join(digit_folder, mfcc_files[0])  # Change index to pick another file
#     mfcc = np.load(mfcc_path)  # Load the MFCC coefficients

#     # Plot the MFCC heatmap for this digit in the corresponding subplot
#     ax = axes[digit]
#     ax.imshow(mfcc, cmap='viridis', aspect='auto', origin='lower')
#     ax.set_title(f"Digit {digit}")
#     ax.set_xlabel('Time')
#     ax.set_ylabel('MFCC Coefficients')

# # Adjust layout for better spacing
# plt.tight_layout()
# plt.show()

interim_data_dir = '../../data/interim/interim_data'  # Folder containing MFCCs

# Function to load the MFCC features for all digits (0-9)
def load_mfcc_data():
    X = []  # Features (MFCC sequences)
    y = []  # Labels

    for digit in range(10):
        digit_folder = os.path.join(interim_data_dir, str(digit))
        mfcc_files = [f for f in os.listdir(digit_folder) if f.endswith('.npy')]
        for mfcc_file in mfcc_files:
            mfcc_path = os.path.join(digit_folder, mfcc_file)
            mfcc = np.load(mfcc_path)
            X.append(mfcc)  # Add sequence to X
            y.append(digit)  # Add label to y

    return X, np.array(y)  # Return sequences and labels without padding

# Load the MFCC data
X, y = load_mfcc_data()

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a dictionary to store the trained HMM models for each digit
hmm_models = {}

# Train HMM models for each digit
for digit in range(10):
    # Get the training data for the current digit
    digit_train_data = [X_train[i] for i in range(len(y_train)) if y_train[i] == digit]
    
    # Concatenate all sequences for the current digit into a single sequence
    # Each sequence is assumed to be of shape (n_frames, n_features), we concatenate along the first axis (time axis)
    digit_train_data_transposed = [sequence.T for sequence in digit_train_data]
    concatenated_sequence = np.concatenate(digit_train_data_transposed, axis=0)
    print(f"Concatenated sequence shape for digit {digit}: {concatenated_sequence.shape}")
    
    # Initialize the HMM model for the current digit
    model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)
    
    # Fit the model to the concatenated sequence
    model.fit(concatenated_sequence)
    
    # Store the trained model for the current digit
    hmm_models[digit] = model

# Function to predict the digit for a given MFCC feature sequence
def predict_digit(mfcc_sequence):
    log_likelihoods = []
    for digit, model in hmm_models.items():
        # Calculate the log likelihood of the sequence for each HMM model
        log_likelihoods.append(model.score(mfcc_sequence.T))
    return np.argmax(log_likelihoods)  # Return the digit with the highest likelihood

# Evaluate the models on the test set
y_pred = []
for i in range(len(X_test)):
    # Predict the digit for each test sample
    predicted_digit = predict_digit(X_test[i])
    y_pred.append(predicted_digit)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Recognition Accuracy: {accuracy * 100:.2f}%")




# Function to extract MFCC features from audio files and convert to numpy array
def extract_mfcc_from_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCC features (13 coefficients)
    # print(f"Extracted MFCC with shape: {mfcc.shape}")
    return mfcc  # Return the MFCC as a NumPy array

# Function to test the model on personal data
def test_personal_data(personal_data_dir):
    y_pred = []
    y_true = []

    for digit in range(10):  # Loop through digits 0 to 9
        digit_folder = os.path.join(personal_data_dir, str(digit))
        
        if not os.path.exists(digit_folder):  # If the folder doesn't exist, skip
            continue
        
        for audio_file in os.listdir(digit_folder):  # Loop through audio files in the digit folder
            if not audio_file.endswith('.wav'):  # Skip non-WAV files
                continue

            file_path = os.path.join(digit_folder, audio_file)

            # Extract MFCC features from the audio file (without saving as .npy)
            mfcc_sequence = extract_mfcc_from_audio(file_path)
            print(f"Extracted MFCC with shape: {mfcc_sequence.T.shape}")
            # Predict the digit using the trained HMM models
            predicted_digit = predict_digit(mfcc_sequence)

            # Store the predictions and true labels
            y_pred.append(predicted_digit)
            y_true.append(digit)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Personal Recordings Accuracy: {accuracy * 100:.2f}%")

# Path to the folder containing personal recordings
personal_data_dir = '../../data/interim/interim_personal_recordings'

# Run the testing loop
test_personal_data(personal_data_dir)