import numpy as np
import torchaudio
import torchaudio.transforms as transforms

def load_audio(file_path):
    """
    Load an audio file and return the waveform and sample rate.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        waveform (Tensor): Audio waveform.
        sample_rate (int): Sample rate of the audio.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def preprocess_audio(waveform, sample_rate, target_sample_rate=16000, target_length=16000):
    """
    Preprocess audio by resampling and truncating or padding to a fixed length.
    
    Args:
        waveform (Tensor): Audio waveform.
        sample_rate (int): Original sample rate.
        target_sample_rate (int): Desired sample rate for the output waveform.
        target_length (int): Desired length of the output waveform in samples.
        
    Returns:
        waveform (Tensor): Preprocessed audio waveform.
    """
    # Resample audio if necessary
    if sample_rate != target_sample_rate:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    
    # Truncate or pad waveform
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.size(1) < target_length:
        pad_length = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    
    return waveform

def extract_features(waveform, sample_rate, n_mfcc=13):
    """
    Extract MFCC features from the audio waveform.
    
    Args:
        waveform (Tensor): Audio waveform.
        sample_rate (int): Sample rate of the audio.
        n_mfcc (int): Number of MFCC features to extract.
        
    Returns:
        mfcc_features (Tensor): Extracted MFCC features.
    """
    mfcc = transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    mfcc_features = mfcc(waveform)
    return mfcc_features

def plot_waveform(waveform, sample_rate, title="Waveform"):
    """
    Plot the waveform of an audio signal.
    
    Args:
        waveform (Tensor): Audio waveform.
        sample_rate (int): Sample rate of the audio.
        title (str): Title of the plot.
    """
    import matplotlib.pyplot as plt
    
    waveform_np = waveform.squeeze().cpu().numpy()
    time = np.arange(waveform_np.shape[0]) / sample_rate
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, waveform_np)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
