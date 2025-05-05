import pytest
import torch
from model_class import ResNet18

def test_evaluate_base_resnet18():
    resnet = ResNet18()

    base_resnet_weight_path = 'weights/resnet.pth'
   

    try:
      
        resnet_state_dict = torch.load(base_resnet_weight_path, map_location="cpu")
        resnet.load_state_dict(resnet_state_dict, strict=True)
       

    except Exception as e:
        pytest.fail(f"Failed to load weights: {e}")

def test_evaluate_aug_resnet18():
    resnet_aug = ResNet18()

    
    augmented_resnet_weight_path = "weights/resnet_aug.pth"
    

    try:
      
       
        resnet_aug_state_dict = torch.load(augmented_resnet_weight_path, map_location="cpu")

       
        resnet_aug.load_state_dict(resnet_aug_state_dict, strict=True)
       

    except Exception as e:
        pytest.fail(f"Failed to load weights: {e}")
