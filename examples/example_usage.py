import torch
from transformers import AutoTokenizer
import librosa
import soundfile as sf
from model import ExciteMeterModel  # Assuming your model class is named ExciteMeterModel
from utils import preprocess_audio  # Assuming you have a preprocessing function

def load_model(model_path):
    """
    Load the trained model from the given path.
    """
    model = ExciteMeterModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode
    return model

def predict_exciting_meter(model, audio_path):
    """
    Predict the exciting meter of an audio file.
    """
    # Load and preprocess the audio file
    audio, sample_rate = librosa.load(audio_path, sr=None)
    preprocessed_audio = preprocess_audio(audio, sample_rate)

    # Convert to tensor
    audio_tensor = torch.tensor(preprocessed_audio).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        prediction = model(audio_tensor)
    
    return prediction.item()

def main():
    # Example paths
    model_path = "path/to/your/model.pth"
    audio_path = "path/to/your/audio.wav"

    # Load the trained model
    model = load_model(model_path)

    # Predict the exciting meter for the audio file
    excitement_score = predict_exciting_meter(model, audio_path)

    print(f"Exciting Meter for '{audio_path}': {excitement_score:.2f}")

if __name__ == "__main__":
    main()
