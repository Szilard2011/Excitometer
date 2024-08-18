import os
import torch
from torch.utils.data import Dataset
from utils import preprocess_single_file

class ExcitometerDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing audio data for the ExcitometerModel.
    """
    def __init__(self, data_dir, target_sample_rate=16000, target_length=16000, n_mfcc=13, transform=None):
        """
        Args:
            data_dir (str): Directory with all the audio files.
            target_sample_rate (int): Desired sample rate for the audio.
            target_length (int): Desired length of the audio in samples.
            n_mfcc (int): Number of MFCC features to extract.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self.n_mfcc = n_mfcc
        self.transform = transform
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses the sample at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            sample (dict): A dictionary containing 'features' and 'label'.
        """
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # Preprocess the audio file to extract features
        features = preprocess_single_file(file_path, self.target_sample_rate, self.target_length, self.n_mfcc)
        
        # Extract the label from the file name (assuming the label is part of the file name)
        label = self.extract_label(file_name)

        sample = {'features': features, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def extract_label(self, file_name):
        """
        Extract the label from the file name. Assumes the label is part of the file name.
        
        Args:
            file_name (str): The name of the audio file.
        
        Returns:
            label (int): The extracted label.
        """
        # Example: Assume the file name is in the format 'class_label_123.wav'
        label_str = file_name.split('_')[1]
        label = int(label_str)  # Convert label to an integer, or modify this based on your specific labeling scheme
        return label

