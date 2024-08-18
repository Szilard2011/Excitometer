import pytest
import torch
from model import ExciteMeterModel  # Adjust the import based on your actual file structure
from torch.testing import assert_allclose

def test_model_architecture():
    """
    Test the architecture of the ExciteMeterModel to ensure it has the correct layers and dimensions.
    """
    model = ExciteMeterModel()

    # Check the type of the model
    assert isinstance(model, ExciteMeterModel), "Model is not of type ExciteMeterModel"

    # Check if model contains expected layers
    assert hasattr(model, 'conv1'), "Model does not have the expected conv1 layer"
    assert hasattr(model, 'rnn'), "Model does not have the expected rnn layer"
    assert hasattr(model, 'fc'), "Model does not have the expected fc layer"

def test_model_forward_pass():
    """
    Test the model's forward pass with a dummy input to ensure it produces output without errors.
    """
    model = ExciteMeterModel()

    # Create a dummy input tensor (e.g., batch size of 1, 1 channel, 16000 samples)
    dummy_input = torch.randn(1, 1, 16000)  # Adjust shape based on your model input

    # Perform a forward pass
    output = model(dummy_input)

    # Check if the output is a tensor with expected shape
    assert isinstance(output, torch.Tensor), "Output should be a torch tensor"
    assert output.shape == torch.Size([1]), "Output shape is incorrect"

def test_model_loading():
    """
    Test loading of a pre-trained model to ensure it loads without errors.
    """
    model_path = 'path/to/your/model.pth'
    model = ExciteMeterModel()
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode

    # Create a dummy input tensor for inference
    dummy_input = torch.randn(1, 1, 16000)
    output = model(dummy_input)

    # Validate the output
    assert isinstance(output, torch.Tensor), "Output should be a torch tensor"
    assert output.shape == torch.Size([1]), "Output shape is incorrect"

def test_model_prediction():
    """
    Test that the model's prediction is within a reasonable range.
    """
    model = ExciteMeterModel()

    # Create a dummy input tensor (e.g., batch size of 1, 1 channel, 16000 samples)
    dummy_input = torch.randn(1, 1, 16000)

    # Perform a forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check if the output is within the expected range
    assert output.item() >= 0.0, "Output should be non-negative"
    assert output.item() <= 1.0, "Output should be less than or equal to 1"

if __name__ == "__main__":
    pytest.main()
