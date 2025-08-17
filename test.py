#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,

import time
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available. Some post-processing features will be disabled.")
from utils import data_loader
from tqdm import tqdm
from utils.metrics import Evaluator
# from network.Net import HANet_v2
from PIL import Image

from network.CGNet import HCGMNet, CGNet
import time
start=time.time()
def test(test_loader, Eva_test, save_path, net, device='cuda'):
    print("Start validation!")

    net.train(False)
    net.eval()
    
    for i, (A, B, mask, filename) in enumerate(tqdm(test_loader, desc="Testing")):
        with torch.no_grad():
            try:
                if device == 'cuda':
                    A = A.cuda()
                    B = B.cuda()
                    Y = mask.cuda()
                else:
                    A = A.cpu()
                    B = B.cpu()
                    Y = mask.cpu()
                
                print(f"Processing batch {i+1}, input shapes: A={A.shape}, B={B.shape}")
                
                # Test-Time Augmentation (TTA) for better performance
                predictions = []
                
                # Original prediction
                preds_orig = net(A, B)
                pred_orig = torch.sigmoid(preds_orig[1])
                predictions.append(pred_orig)
                
                # Horizontal flip TTA
                A_hflip = torch.flip(A, dims=[3])
                B_hflip = torch.flip(B, dims=[3])
                preds_hflip = net(A_hflip, B_hflip)
                pred_hflip = torch.flip(torch.sigmoid(preds_hflip[1]), dims=[3])
                predictions.append(pred_hflip)
                
                # Vertical flip TTA
                A_vflip = torch.flip(A, dims=[2])
                B_vflip = torch.flip(B, dims=[2])
                preds_vflip = net(A_vflip, B_vflip)
                pred_vflip = torch.flip(torch.sigmoid(preds_vflip[1]), dims=[2])
                predictions.append(pred_vflip)
                
                # Average all predictions for better performance
                output = torch.mean(torch.stack(predictions), dim=0)
                print(f"TTA ensemble output shape: {output.shape}")
                
                # Apply Otsu's thresholding for optimal threshold per image
                pred = np.zeros_like(output.cpu().numpy(), dtype=int)
                for b in range(output.shape[0]):
                    out_np = output[b, 0].cpu().numpy()
                    
                    # Convert to uint8 for Otsu
                    out_uint8 = (out_np * 255).astype(np.uint8)
                    
                    # Apply Otsu's thresholding for optimal threshold
                    try:
                        import cv2
                        _, otsu_mask = cv2.threshold(out_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        pred[b, 0] = (otsu_mask > 127).astype(int)
                    except ImportError:
                        # Fallback: Use a slightly lower threshold for better recall
                        pred[b, 0] = (out_np >= 0.45).astype(int)
                
                target = Y.cpu().numpy().astype(int)

                for j in range(output.shape[0]):
                    # Apply morphological post-processing for cleaner results
                    try:
                        import cv2
                        mask_bin = (pred[j, 0] * 255).astype(np.uint8)
                        
                        # Morphological operations to clean up the mask
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        
                        # Close small gaps first
                        mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
                        
                        # Remove small noise
                        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
                        
                        # Remove very small components
                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_opened, connectivity=8)
                        mask_filtered = np.zeros_like(mask_opened)
                        
                        # Keep components larger than 50 pixels
                        min_area = 50
                        for label in range(1, num_labels):
                            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                                mask_filtered[labels == label] = 255
                        
                        final_mask = mask_filtered
                        # Update the prediction array with filtered result
                        pred[j, 0] = (final_mask > 127).astype(int)
                        
                    except ImportError:
                        # If OpenCV not available, use the original prediction
                        final_mask = (pred[j, 0] * 255).astype(np.uint8)
                    
                    # Save prediction
                    final_savepath = save_path + filename[j] + '.png'
                    if not os.path.exists(os.path.dirname(final_savepath)):
                        os.makedirs(os.path.dirname(final_savepath))
                    
                    im = Image.fromarray(final_mask)
                    im.save(final_savepath)
                    print(f"Saved prediction: {final_savepath}")

                Eva_test.add_batch(target, pred)
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                break

    IoU = Eva_test.Intersection_over_Union()
    Pre = Eva_test.Precision()
    Recall = Eva_test.Recall()
    F1 = Eva_test.F1()
    OA=Eva_test.OA()
    Kappa=Eva_test.Kappa()

    # print('[Test] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    print('[Test] F1: %.4f, Precision:%.4f, Recall: %.4f, OA: %.4f, Kappa: %.4f,IoU: %.4f' % ( F1[1],Pre[1],Recall[1],OA[1],Kappa[1],IoU[1]))
    # print('F1-Score: {:.2f}\nPrecision: {:.2f}\nRecall: {:.2f}\nOA: {:.2f}\nKappa: {:.2f}\nIoU: {:.2f}\n}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100, IoU[1] * 100))
    print('F1-Score: Precision: Recall: OA: Kappa: IoU: ')
    print('{:.2f}\\{:.2f}\\{:.2f}\\{:.2f}\\{:.2f}\\{:.2f}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100,IoU[1] * 100))
    print('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100,IoU[1] * 100))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--data_name', type=str, default='', 
                        help='the test dataset name (leave empty for interactive selection)')
    parser.add_argument('--model_name', type=str, default='CGNet',
                        help='the model name')
    parser.add_argument('--save_path', type=str,
                        default='./test_result/')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    if opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    if opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    # --- Interactive Dataset Selection ---
    dataset_base_dir = "dataset"
    if not opt.data_name:  # If no dataset specified via command line
        available_datasets = [name for name in os.listdir(dataset_base_dir) 
                            if os.path.isdir(os.path.join(dataset_base_dir, name))]
        
        if not available_datasets:
            print(f"No datasets found in {dataset_base_dir}. Please ensure datasets are placed in subdirectories.")
            exit(1)

        print("\nAvailable datasets for testing:")
        for i, ds_name in enumerate(available_datasets):
            test_path = os.path.join(dataset_base_dir, ds_name, 'test')
            if os.path.exists(test_path):
                # Count number of test images
                try:
                    a_path = os.path.join(test_path, 'A')
                    if os.path.exists(a_path):
                        num_images = len([f for f in os.listdir(a_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                        print(f"{i+1}. {ds_name} ({num_images} test images)")
                    else:
                        print(f"{i+1}. {ds_name} (test folder structure unclear)")
                except:
                    print(f"{i+1}. {ds_name}")
            else:
                print(f"{i+1}. {ds_name} (no test folder)")
        
        selected_dataset_idx = -1
        while selected_dataset_idx < 0 or selected_dataset_idx >= len(available_datasets):
            try:
                choice = int(input(f"\nSelect dataset to test (1-{len(available_datasets)}): ")) - 1
                if 0 <= choice < len(available_datasets):
                    selected_dataset_idx = choice
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        opt.data_name = available_datasets[selected_dataset_idx]
        print(f"Selected dataset: {opt.data_name}")
    else:
        print(f"Using dataset specified via command line: {opt.data_name}")
    # --- End Interactive Dataset Selection ---

    # --- Interactive Model Selection ---
    if opt.model_name == 'CGNet':  # Default, allow user to change
        print("\nAvailable models:")
        print("1. CGNet (Change Guiding Network)")
        print("2. HCGMNet (Hierarchical Change Guiding Map Network)")
        
        model_choice = ""
        while model_choice not in ["1", "2"]:
            model_choice = input("\nSelect model to use (1-CGNet, 2-HCGMNet): ")
            if model_choice not in ["1", "2"]:
                print("Invalid selection. Please try again.")
        
        if model_choice == "1":
            opt.model_name = "CGNet"
        else:
            opt.model_name = "HCGMNet"
        
        print(f"Selected model: {opt.model_name}")
    else:
        print(f"Using model specified via command line: {opt.model_name}")
    # --- End Interactive Model Selection ---

    # Set test root based on dataset
    dataset_base_dir = "dataset"
    opt.test_root = os.path.join(dataset_base_dir, opt.data_name, 'test')
    
    print(f"Looking for test dataset at: {opt.test_root}")
    if not os.path.exists(opt.test_root):
        print(f"Error: Test dataset path not found: {opt.test_root}")
        print("Available datasets:")
        if os.path.exists(dataset_base_dir):
            for item in os.listdir(dataset_base_dir):
                if os.path.isdir(os.path.join(dataset_base_dir, item)):
                    print(f"  - {item}")
        exit()
    else:
        print(f"Found test dataset at: {opt.test_root}")

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name + '/'
    print(f"Loading test data from: {opt.test_root}")
    test_loader = data_loader.get_test_loader(opt.test_root, opt.batchsize, opt.trainsize, num_workers=0, shuffle=False, pin_memory=False)
    print(f"Test data loaded successfully. Number of batches: {len(test_loader)}")
    Eva_test = Evaluator(num_class=2)

    if opt.model_name == 'HCGMNet':
        print("Initializing HCGMNet model...")
        model = HCGMNet(backbone_name='resnet34')
    elif opt.model_name == 'CGNet':
        print("Initializing CGNet model...")
        model = CGNet(backbone_name='resnet34', use_aspp=False)
    
    # Use CPU only for now to avoid CUDA issues
    print("Using CPU for inference (CUDA seems to have issues)")
    device = 'cpu'
    
    print(f"Model {opt.model_name} initialized successfully on {device}")

    opt.load = './output/' + opt.data_name + '/' + opt.model_name + '-resnet34_best_iou.pth'
    print(f"Looking for model weights at: {opt.load}")

    # If the default path doesn't exist, try to find any available model weights
    if not os.path.exists(opt.load):
        output_dir = os.path.join('./output', opt.data_name)
        if os.path.exists(output_dir):
            available_models = [f for f in os.listdir(output_dir) if f.endswith('.pth') and opt.model_name in f]
            if available_models:
                print(f"Default model not found. Available models for {opt.model_name}:")
                for i, model_file in enumerate(available_models):
                    print(f"{i+1}. {model_file}")
                
                if len(available_models) == 1:
                    opt.load = os.path.join(output_dir, available_models[0])
                    print(f"Auto-selected: {available_models[0]}")
                else:
                    model_choice = -1
                    while model_choice < 0 or model_choice >= len(available_models):
                        try:
                            choice = int(input(f"\nSelect model weights (1-{len(available_models)}): ")) - 1
                            if 0 <= choice < len(available_models):
                                model_choice = choice
                            else:
                                print("Invalid selection. Please try again.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                    
                    opt.load = os.path.join(output_dir, available_models[model_choice])
                    print(f"Selected: {available_models[model_choice]}")

    if opt.load is not None and os.path.exists(opt.load):
        print(f"Loading model weights from: {opt.load}")
        try:
            # Load weights to CPU to avoid CUDA issues
            state_dict = torch.load(opt.load, map_location='cpu')
            model.load_state_dict(state_dict)
            print('Model loaded successfully from', opt.load)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()
    else:
        print(f"Error: Model weights not found at {opt.load}")
        print("Available model files in output directory:")
        output_dir = os.path.dirname(opt.load)
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith('.pth'):
                    print(f"  - {file}")
        else:
            print(f"Output directory doesn't exist: {output_dir}")
        exit()

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    test(test_loader, Eva_test, opt.save_path, model, device)

end=time.time()
print('Time:',end-start)