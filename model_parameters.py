import torch
from network.CGNet import CGNet, HCGMNet
from prettytable import PrettyTable

def count_parameters(model):
    """Count number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model_parameters(model):
    """Analyze parameters for each layer/module in the model"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    return table, total_params

def print_model_analysis(model_name, backbone_name):
    """Print detailed analysis for a specific model and backbone"""
    print(f"\n{'='*80}")
    print(f"Analyzing {model_name} with {backbone_name} backbone")
    print('='*80)
    
    # Initialize model
    if model_name == "CGNet":
        model = CGNet(backbone_name=backbone_name)
    elif model_name == "HCGMNet":
        model = HCGMNet(backbone_name=backbone_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get parameter table and total count
    table, total_params = analyze_model_parameters(model)
    
    # Print results
    print(table)
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    print(f"Model Size (MB): {total_params * 4 / (1024*1024):.2f}")
    print('='*80 + '\n')
    
    return total_params

def main():
    # List of models and backbones to analyze
    models = ["CGNet", "HCGMNet"]
    backbones = ["resnet34", "vgg16"]
    
    # Store results for comparison
    results = {}
    
    # Analyze each combination
    for model_name in models:
        results[model_name] = {}
        for backbone in backbones:
            params = print_model_analysis(model_name, backbone)
            results[model_name][backbone] = params
    
    # Print comparison summary
    print("\nModel Comparison Summary:")
    print('='*80)
    table = PrettyTable()
    table.field_names = ["Model", "Backbone", "Parameters", "Size (MB)"]
    
    for model_name in models:
        for backbone in backbones:
            params = results[model_name][backbone]
            table.add_row([
                model_name,
                backbone,
                f"{params:,}",
                f"{params * 4 / (1024*1024):.2f}"
            ])
    
    print(table)

if __name__ == "__main__":
    main() 