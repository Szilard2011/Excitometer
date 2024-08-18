import os
import torch
import torchaudio
import numpy as np
from utils import load_audio, preprocess_audio, extract_features

def preprocess_dataset(input_dir, output_dir, target_sample_rate=16000, target_length=16000, n_mfcc=13):
    """
    Preprocess the entire dataset by loading, processing, and saving the audio files.
    
    Args:
        input_dir (str): Directory containing the raw audio files.
        output_dir (str): Directory where the processed files will be saved.
        target_sample_rate (int): Desired sample rate for the audio.
        target_length (int): Desired length of the audio in samples.
        n_mfcc (int): Number of MFCC features to extract.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            # Load audio
            file_path = os.path.join(input_dir, file_name)
            waveform, sample_rate = load_audio(file_path)
            
            # Preprocess audio
            waveform = preprocess_audio(waveform, sample_rate, target_sample_rate, target_length)
            
            # Extract features
            features = extract_features(waveform, target_sample_rate, n_mfcc)
            
            # Save processed features
            output_path = os.path.join(output_dir, file_name.replace('.wav', '.pt'))
            torch.save(features, output_path)
            print(f"Processed and saved {file_name} to {output_path}")

def preprocess_single_file(file_path, target_sample_rate=16000, target_length=16000, n_mfcc=13):
    """
    Preprocess a single audio file and return the processed features.
    
    Args:
        file_path (str): Path to the audio file.
        target_sample_rate (int): Desired sample rate for the audio.
        target_length (int): Desired length of the audio in samples.
        n_mfcc (int): Number of MFCC features to extract.
        
    Returns:
        features (Tensor): Preprocessed and feature-extracted audio.
    """
    # Load audio
    waveform, sample_rate = load_audio(file_path)
    
    # Preprocess audio
    waveform = preprocess_audio(waveform, sample_rate, target_sample_rate, target_length)
    
    # Extract features
    features = extract_features(waveform, target_sample_rate, n_mfcc)
    
    return features

# Example usage:
if __name__ == "__main__":
    input_dir = "path/to/raw/audio"
    output_dir = "path/to/processed/audio"
    
    # Preprocess the entire dataset
    preprocess_dataset(input_dir, output_dir)
    
    # Preprocess a single file (for testing)
    # features = preprocess_single_file("path/to/single/audio/file.wav")
    # print(features.shape)
