import torch
import pytest
from model_class import SegNet_Encoder, SegNet_Decoder, SegNet_Pretrained,DeepLabV3
import torch.nn as nn
import torch.nn.functional as F






def test_evaluate_segnet():
    # Create an instance of SegNet_Trained
    segnet_model = SegNet_Pretrained('encoder_model.pth')
    
    # Create a sample input tensor
    x = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32

    # Forward pass
    output = segnet_model.forward(x)

    print("Output shape:", output.shape)

    try:
        assert (output.shape == torch.Size([1, 32, 32, 32]))
    except:
        pytest.fail(f"Code snippet raised an exception")
def test_evaluate_deeplabv3():
    model=DeepLabV3(32)
    x = torch.randn(3, 3, 512, 512)  # batch_size=1, channels=3, height=32, width=32

    # Forward pass
    model.train()
    output = model.forward(x)

    print("Output shape:", output.shape)

    try:
        assert (output.shape == torch.Size([3, 32,512, 512]))
    except:
        pytest.fail(f"Code snippet raised an exception")