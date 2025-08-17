import torch
from thop import profile
import os
from network.CGNet import CGNet, HCGMNet

def profile_model(model, input_size):
    """
    Profiles the given model to calculate FLOPs and parameters.
    """
    # Create dummy inputs
    input1 = torch.randn(1, 3, input_size, input_size)
    input2 = torch.randn(1, 3, input_size, input_size)
    
    # Use thop to profile
    macs, params = profile(model, inputs=(input1, input2), verbose=False)
    
    # GFLOPs is typically calculated from MACs. 1 MAC = 2 FLOPs approx.
    gflops = macs * 2 / 1e9
    
    # Parameters are in millions
    params_m = params / 1e6
    
    return gflops, params_m

if __name__ == '__main__':
    # --- Interactive Setup ---
    # 1. Model Selection
    model_choices_map = {"1": "CGNet", "2": "HCGMNet"}
    print("\nSelect model architecture:")
    print("1. CGNet")
    print("2. HCGMNet")
    
    selected_model_name = ""
    while selected_model_name not in model_choices_map.values():
        choice = input("Select model (1-2): ")
        selected_model_name = model_choices_map.get(choice)
        if not selected_model_name:
            print("Invalid selection. Please try again.")
    print(f"Selected model: {selected_model_name}")

    # 2. Backbone Selection
    backbone_choices_map = {"1": "vgg16", "2": "resnet34"}
    print("\nSelect backbone architecture:")
    print("1. VGG16")
    print("2. ResNet34")
    
    selected_backbone_name = ""
    while selected_backbone_name not in backbone_choices_map.values():
        choice = input("Select backbone (1-2): ")
        selected_backbone_name = backbone_choices_map.get(choice)
        if not selected_backbone_name:
            print("Invalid selection. Please try again.")
    print(f"Selected backbone: {selected_backbone_name}")

    # 3. Input Size
    input_size = 0
    while input_size <= 0:
        try:
            size_str = input("\nEnter the input size for the model (e.g., 256 for a 256x256 image): ")
            input_size = int(size_str)
            if input_size <= 0:
                print("Input size must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    print(f"Using input size: {input_size}x{input_size}")
    
    # --- Model Instantiation and Profiling ---
    print("\nInstantiating model...")
    try:
        if selected_model_name == 'CGNet':
            model = CGNet(backbone_name=selected_backbone_name)
        elif selected_model_name == 'HCGMNet':
            model = HCGMNet(backbone_name=selected_backbone_name)
        else:
            # This case should not be reached due to the interactive prompt
            raise ValueError(f"Unknown model name: {selected_model_name}")
            
        print("Model instantiated successfully.")
        
        print("Profiling model...")
        gflops, params_m = profile_model(model, input_size)
        
        print("\n--- Model Profile Report ---")
        print(f"  Model:          {selected_model_name}")
        print(f"  Backbone:       {selected_backbone_name}")
        print(f"  Input Size:     {input_size}x{input_size}")
        print("------------------------------")
        print(f"  GFLOPs:         {gflops:.2f} G")
        print(f"  Parameters:     {params_m:.2f} M")
        print("------------------------------\n")
        
    except Exception as e:
        print(f"\nAn error occurred during model instantiation or profiling: {e}")
        print("Please ensure the backbone and model names are correct and the environment is set up properly.") 