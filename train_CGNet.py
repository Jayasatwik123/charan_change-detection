#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import optim

import utils.visualization as visual
from utils import data_loader
from torch.optim import lr_scheduler
from tqdm import tqdm
import random
from utils.utils import clip_gradient, adjust_lr
from utils.metrics import Evaluator

from utils.boundary_loss import BoundaryLoss


# --- BCEWithLogitsLoss with Label Smoothing ---
class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        # Apply label smoothing: y_smooth = y * (1 - smoothing) + 0.5 * smoothing
        targets = targets.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# --- Dice Loss Implementation ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
import glob

from network.CGNet import HCGMNet,CGNet

import time
start=time.time()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"GPU Memory Free: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB used\n")
    else:
        print("No CUDA GPU available!")

def train(train_loader, val_loader, Eva_train, Eva_val, current_epoch, total_epochs, model_save_path, net, criterion, optimizer, vis, label_smoothing=0.1):
    global best_iou, g_best_f1_for_best_iou, g_best_epoch_num
    epoch_loss = 0
    net.train(True)

    length = 0
    boundary_loss_fn = BoundaryLoss()
    dice_loss_fn = DiceLoss()
    focal_loss_fn = FocalLoss()
    bce_smooth_fn = BCEWithLogitsLossWithLabelSmoothing(smoothing=label_smoothing)
    for i, (A, B, mask) in enumerate(tqdm(train_loader)):
        A = A.cuda()
        B = B.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()
        preds = net(A,B)
        # Main loss (BCE with label smoothing + Dice + Focal)
        bce_loss = bce_smooth_fn(preds[0], Y) + bce_smooth_fn(preds[1], Y)
        dice_loss = dice_loss_fn(preds[0], Y) + dice_loss_fn(preds[1], Y)
        focal_loss = focal_loss_fn(preds[0], Y) + focal_loss_fn(preds[1], Y)
        # Boundary loss (only on final output)
        b_loss = boundary_loss_fn(preds[1], Y)
        # You can tune the weights below
        loss = bce_loss + dice_loss + 0.5 * focal_loss + 0.5 * b_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        output = torch.sigmoid(preds[1])
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        pred = output.data.cpu().numpy().astype(int)
        target = Y.cpu().numpy().astype(int)
        
        Eva_train.add_batch(target, pred)

        length += 1
    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(current_epoch, IoU, 'mIoU')
    vis.add_scalar(current_epoch, Pre, 'Precision')
    vis.add_scalar(current_epoch, Recall, 'Recall')
    vis.add_scalar(current_epoch, F1, 'F1')
    vis.add_scalar(current_epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            current_epoch, total_epochs, \
            train_loss, \
            IoU, Pre, Recall, F1))
    print("Strat validing!")


    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A,B)[1]
            output = torch.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_val.add_batch(target, pred)

            length += 1
    val_iou_metrics = Eva_val.Intersection_over_Union()
    val_pre_metrics = Eva_val.Precision()
    val_recall_metrics = Eva_val.Recall()
    val_f1_metrics = Eva_val.F1()

    current_val_iou = val_iou_metrics[1]
    current_val_precision = val_pre_metrics[1]
    current_val_recall = val_recall_metrics[1]
    current_val_f1 = val_f1_metrics[1]

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (current_val_iou, current_val_precision, current_val_recall, current_val_f1))
    
    if current_val_iou >= best_iou:
        best_iou = current_val_iou
        g_best_f1_for_best_iou = current_val_f1
        g_best_epoch_num = current_epoch
        
        best_net = net.state_dict()
        print('New Best Model Found & Saved! IoU: %.4f; F1: %.4f; Precision: %.4f; Recall: %.4f; Epoch: %d' % 
              (best_iou, g_best_f1_for_best_iou, current_val_precision, current_val_recall, g_best_epoch_num))
        torch.save(best_net, model_save_path + '_best_iou.pth')
    
    print('Overall Best IoU so far: %.4f (Achieved at Epoch %d with F1: %.4f)' % (best_iou, g_best_epoch_num, g_best_f1_for_best_iou))


if __name__ == '__main__':
    seed_everything(42)
    import argparse
    
    # Clear CUDA cache and check GPU
    torch.cuda.empty_cache()
    print_gpu_info()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--trainsize', type=int, default=256, help='training image size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--model_name', type=str, default='CGNet', choices=['CGNet', 'HCGMNet'], help='Model class to use (CGNet or HCGMNet)')
    parser.add_argument('--save_path_base', type=str, default='./output/', help='base directory for saving models')
    parser.add_argument('--use_aspp', action='store_true', help='Enable ASPP module in CGNet (ignored for HCGMNet)')
    opt = parser.parse_args()

    # --- Interactive Setup ---
    # 1. Dataset Selection
    dataset_base_dir = "dataset/"
    available_datasets = [name for name in os.listdir(dataset_base_dir) if os.path.isdir(os.path.join(dataset_base_dir, name))]
    
    if not available_datasets:
        print(f"No datasets found in {dataset_base_dir}. Please ensure datasets are placed in subdirectories.")
        exit(1)

    print("\nAvailable datasets:")
    for i, ds_name in enumerate(available_datasets):
        print(f"{i+1}. {ds_name}")
    
    selected_dataset_idx = -1
    while selected_dataset_idx < 0 or selected_dataset_idx >= len(available_datasets):
        try:
            choice = int(input(f"Select dataset (1-{len(available_datasets)}): ")) - 1
            if 0 <= choice < len(available_datasets):
                selected_dataset_idx = choice
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    selected_dataset_name = available_datasets[selected_dataset_idx]
    print(f"Selected dataset: {selected_dataset_name}")

    opt.train_root = os.path.join(dataset_base_dir, selected_dataset_name, "train/")
    opt.val_root = os.path.join(dataset_base_dir, selected_dataset_name, "val/")

    if not os.path.isdir(opt.train_root) or not os.path.isdir(opt.val_root):
        print(f"Error: 'train' or 'val' subdirectory not found in {os.path.join(dataset_base_dir, selected_dataset_name)}")
        print(f"Please ensure {opt.train_root} and {opt.val_root} exist.")
        exit(1)

    # 2. Batch Size
    interactive_batch_size = -1
    while interactive_batch_size <= 0:
        try:
            interactive_batch_size = int(input("Enter batch size (e.g., 8): "))
            if interactive_batch_size <= 0:
                print("Batch size must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer for batch size.")
    opt.batchsize = interactive_batch_size
    print(f"Using batch size: {opt.batchsize}")

    # 3. Backbone Selection
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

    # 4. ASPP Selection (only for CGNet)
    use_aspp = False
    if opt.model_name == 'CGNet':
        print("\nEnable ASPP module? (Recommended for multi-scale context)")
        print("1. Yes (with ASPP)")
        print("2. No (vanilla CGNet)")
        aspp_choice = ""
        while aspp_choice not in ["1", "2"]:
            aspp_choice = input("Enable ASPP? (1-Yes, 2-No): ")
            if aspp_choice not in ["1", "2"]:
                print("Invalid selection. Please try again.")
        use_aspp = (aspp_choice == "1")
    else:
        use_aspp = False
    print(f"ASPP enabled: {use_aspp}")
    # --- End Interactive Setup ---

    # set the device for training
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("ERROR: CUDA is not available. Please install CUDA and appropriate drivers.")
        exit(1)
        
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id 
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print(f"Training will use GPU: {torch.cuda.get_device_name(0)} (selected via CUDA_VISIBLE_DEVICES='{opt.gpu_id}')")
    else:
        print(f"ERROR: CUDA is not available for PyTorch after setting CUDA_VISIBLE_DEVICES='{opt.gpu_id}'. Please check your CUDA setup and GPU ID.")
        exit(1)
    
    # Construct save_path based on choices
    aspp_tag = "_ASPP" if (opt.model_name == 'CGNet' and use_aspp) else ""
    model_save_path_final = os.path.join(opt.save_path_base, selected_dataset_name, f"{opt.model_name}-{selected_backbone_name}{aspp_tag}")
    os.makedirs(model_save_path_final, exist_ok=True)
    
    train_loader = data_loader.get_loader(opt.train_root, opt.batchsize, opt.trainsize, num_workers=0, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=0, shuffle=False, pin_memory=True)
    Eva_train = Evaluator(num_class = 2)
    Eva_val = Evaluator(num_class=2)

    # Initialize visualization object once
    vis = visual.Visualization()
    tensorboard_log_name = f"{opt.model_name}_{selected_backbone_name}_{selected_dataset_name}"
    vis.create_summary(model_type=tensorboard_log_name)


    # Model Instantiation with selected backbone
    if opt.model_name == 'HCGMNet':
        model = HCGMNet(backbone_name=selected_backbone_name).to(device)
    elif opt.model_name == 'CGNet':
        model = CGNet(backbone_name=selected_backbone_name, use_aspp=use_aspp).to(device)
    else: # Should not happen due to choices in argparse
        print(f"Error: Unknown model_name '{opt.model_name}'")
        exit(1)


    # Use BCEWithLogitsLoss with label smoothing
    criterion = nn.BCEWithLogitsLoss().to(device)  # For validation, keep original
    label_smoothing = 0.1  # You can tune this value (0.05-0.2 typical)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # Initialize best metrics tracking
    best_iou = 0.0
    g_best_f1_for_best_iou = 0.0
    g_best_epoch_num = 0

    print("Start train...")
    for epoch_iter in range(1, opt.epoch + 1):
        current_lr = adjust_lr(optimizer, opt.lr, epoch_iter, opt.decay_rate, opt.decay_epoch)
        print(f"\nEpoch {epoch_iter}/{opt.epoch}")
        print(f"Current LR: {current_lr}")

        Eva_train.reset()
        Eva_val.reset()
        train(train_loader, val_loader, Eva_train, Eva_val, epoch_iter, opt.epoch, model_save_path_final, model, criterion, optimizer, vis, label_smoothing=label_smoothing)
        lr_scheduler.step()
    
    vis.close_summary()

    end = time.time()
    print('Running time: %s Seconds' % (end - start))