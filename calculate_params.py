import torch
from network.CGNet import CGNet
from utils.utils import CalParams
import os
import sys

# Define the model path
model_path = r"C:\Users\Cynix\Desktop\CGNet_workingOnBackBone_ssd\output\LEVIR-CD-256\CGNet-resnet34_best_iou.pth"

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

# Set device (CPU is fine for parameter counting)
device = torch.device("cpu")

# 1. Instantiate the CGNet model with the ResNet-34 backbone
try:
    print("Instantiating CGNet model with ResNet-34 backbone...")
    model = CGNet(backbone_name='resnet34').to(device)
    model.eval() # Set to evaluation mode

    # 2. Load the trained weights
    print(f"Loading trained weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle potential DataParallel wrapper
    if all(key.startswith('module.') for key in state_dict.keys()):
        print("Stripping 'module.' prefix from state_dict keys...")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    print(f"Successfully loaded model from {model_path}")

    # 3. Create dummy input tensors (batch_size, channels, height, width)
    dummy_input_A = torch.randn(1, 3, 256, 256).to(device)
    dummy_input_B = torch.randn(1, 3, 256, 256).to(device)
    print("Created dummy input tensors.")

    # 4. Use CalParams to count parameters and FLOPs
    print("\nCalculating parameters and FLOPs...")
    CalParams(model, (dummy_input_A, dummy_input_B))

except FileNotFoundError:
    print(f"Error: Could not find the model file at {model_path}")
    sys.exit(1)
except ImportError as e:
    print(f"ImportError: {e}. Please ensure PyTorch and other dependencies are installed and accessible.")
    print("Also check if network.CGNet and utils.utils are in the Python path or current directory.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure that the network.CGNet and utils.utils modules are accessible,")
    print("the model path is correct, and the model was saved in a compatible way.")
    sys.exit(1) 