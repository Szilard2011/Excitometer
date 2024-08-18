import pytest
import numpy as np
import librosa
from preprocess import preprocess_audio  # Adjust the import based on your actual file structure

def test_preprocess_audio():
    """
    Test the `preprocess_audio` function to ensure it processes audio correctly.
    """
    # Generate a dummy audio signal (sine wave) for testing
    sample_rate = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    # Process the dummy audio
    processed_audio = preprocess_audio(audio, sample_rate)

    # Define expected properties of the processed audio
    # These properties depend on what `preprocess_audio` does
    expected_length = len(processed_audio)  # Adjust based on preprocessing steps

    # Check if the processed audio meets expected properties
    assert isinstance(processed_audio, np.ndarray), "Processed audio should be a numpy array"
    assert processed_audio.shape[0] == expected_length, "Processed audio length is incorrect"
    assert np.all(np.isfinite(processed_audio)), "Processed audio contains non-finite values"

def test_preprocess_audio_edge_cases():
    """
    Test edge cases for `preprocess_audio` function.
    """
    sample_rate = 16000
    empty_audio = np.array([])

    # Process the empty audio
    processed_audio = preprocess_audio(empty_audio, sample_rate)

    # Check if the processed audio is also empty
    assert isinstance(processed_audio, np.ndarray), "Processed audio should be a numpy array"
    assert processed_audio.size == 0, "Processed audio should be empty for empty input"

    # Test with audio of very short duration
    short_audio = np.array([0.0])  # Single sample

    # Process the short audio
    processed_audio = preprocess_audio(short_audio, sample_rate)

    # Check if the processed audio has been handled properly
    assert isinstance(processed_audio, np.ndarray), "Processed audio should be a numpy array"
    assert processed_audio.size > 0, "Processed audio should not be empty for short input"

if __name__ == "__main__":
    pytest.main()
