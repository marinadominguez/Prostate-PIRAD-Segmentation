import sys
import os

# Add the project directory (HOSP_pipeline) to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb
from HOSP_pipeline.models.unet3d import UNet3D
from HOSP_pipeline.training.transforms import train_transform_cuda, val_transform_cuda
from HOSP_pipeline.preprocessing.dataset_hosp import Prostate3DDataset, get_train_val_test_Dataloaders
from HOSP_pipeline.utils.utils import resize_image_and_label
import HOSP_pipeline.config.config as cfg
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from HOSP_pipeline.config.config import config
from HOSP_pipeline.config.config import (TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS,  
                    TRAIN_CUDA, SAVE_INTERVAL, SAVE_CHECKPOINT_PATH, BACKGROUND_AS_CLASS, 
                    GAMMA, ALPHA)
from HOSP_pipeline.training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
from HOSP_pipeline.training.my_focal_loss import CustomFocalLoss, CustomFocalForegroundDiceLoss, CustomFocalLoss4finetuning



name_model = 'fine_tuned_model_5-5_alpha0.7' #soft_masking
wandb.init(project="hops_pipeline", entity="marina-dominguez", name=name_model)
# Log the entire config dictionary
wandb.config.update(config)
# Now you can access it like this:
print("Dataset path:", wandb.config.dataset_path)
print("Number of epochs:", wandb.config.training_epoch)

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

# Load the pre-trained model
model = UNet3D(in_channels=cfg.IN_CHANNELS, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH), strict=False)  # Path to your pre-trained model

# Freeze all layers except the synthesis path (up-convolutions)
for name, param in model.named_parameters():
    if "s_block" in name:
        param.requires_grad = True  # Unfreeze synthesis path layers
    else:
        param.requires_grad = False  # Freeze other layers

# Move the model to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define optimizer with a lower learning rate for fine-tuning
optimizer = Adam(params=model.parameters(), lr=1e-4)

# Define the loss function using BCEWithLogitsLoss
# criterion = nn.BCEWithLogitsLoss().cuda()

# Initialize DataLoaders, transformations & criterion (depending on whether cuda is available)
if torch.cuda.is_available() and TRAIN_CUDA:
    device = torch.device('cuda')
    model = model.to(device)
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
    criterion = CustomFocalLoss4finetuning(alpha=ALPHA, gamma=GAMMA, reduction='mean').cuda()
    print('cuda available! Training initialized on gpu ...')

elif not torch.cuda.is_available() and TRAIN_CUDA:
    #criterion = CustomFocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    criterion = CustomFocalLoss4finetuning(alpha=ALPHA, gamma=GAMMA, reduction='mean')
    train_transforms = train_transform
    val_transforms = val_transform
    print('cuda not available! Training initialized on cpu ...')

# Load your new hospital dataset
train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms=train_transform_cuda, 
                                                                     val_transforms=val_transform_cuda, 
                                                                     test_transforms=val_transform_cuda)
def dice_coef(pred, target, smooth=1e-5):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice
# Apply filtering to the dataset
#train_dataloader.dataset.data = filter_patients_and_slices(train_dataloader.dataset.data)
#val_dataloader.dataset.data = filter_patients_and_slices(val_dataloader.dataset.data)
"""
# Soft masking strategy based on slice indices
def apply_soft_masking(slice_idx, label):
    # Define ranges for core and edge slices based on the statistical analysis
    core_slices_range = range(10, 21)  # Adjust based on your histogram
    edge_slices_range = range(5, 10) + range(21, 26)

    if slice_idx in core_slices_range:
        # Strict masking (full penalty)
        mask = (label > 0).float()
    elif slice_idx in edge_slices_range:
        # Soft masking (reduced penalty)
        mask = (label > 0).float() * 0.5  # Adjust the weight as needed
    else:
        # No penalty for slices outside likely prostate range
        mask = torch.zeros_like(label).float()
    
    return mask"""

# Fine-tuning loop
for epoch in range(5):  # Fine-tune for a smaller number of epochs
    print(f"Epoch {epoch + 1}/5")
    model.train()
    train_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = 4  # Accumulate gradients over 4 steps
    
    for i, data in enumerate(train_dataloader):
        print(f"Batch {i + 1}/{len(train_dataloader)}")

        images = data[0].to(device)
        labels = data[1].to(device)

        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
        image, ground_truth = resize_image_and_label(images, labels)
        image = image.float()
        ground_truth = ground_truth.long()

        # Extract slice index from the 'slice' field (assuming it's included)
        # slice_idx = int(data[2][0].split('_')[-1].replace('.dcm', ''))  # Adjust based on your data structure

        # Apply appropriate masking
        # mask = apply_soft_masking(slice_idx, ground_truth)

        # Only zero the gradients at the start of accumulation
        if i % accumulation_steps == 0:
            optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(image)  # Do not apply sigmoid here I don't understand wwhy so i'm doing it
            output = torch.sigmoid(output)
            ground_truth = ground_truth.squeeze(1)
            one_hot_gt = F.one_hot(ground_truth, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            loss = criterion(output, one_hot_gt) / accumulation_steps# , mask) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # Reset gradients for the next accumulation

        train_loss += loss.item()
        torch.cuda.empty_cache()
    
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Avg Train Loss: {avg_train_loss}")
    wandb.log({"Avg Train Loss": avg_train_loss})

    # Validation (same as original code)
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for data in val_dataloader:

            images = data[0].to(device)
            labels = data[1].to(device)

            image, ground_truth = resize_image_and_label(images, labels)
            image = image.float()
            ground_truth = ground_truth.long()

            #slice_idx = data[2][0].split('_')[-1].replace('.dcm', '')  # Adjust based on your data structure
            #slice_idx = int(slice_idx.split('/')[-1])
            # mask = apply_soft_masking(slice_idx, ground_truth)

            output = model(image)  # Do not apply sigmoid here
            output = torch.sigmoid(output)
            ground_truth = ground_truth.squeeze(1)
            one_hot_gt = F.one_hot(ground_truth, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()

            loss = criterion(output, one_hot_gt)#, mask)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Avg Val Loss: {avg_val_loss}")
    wandb.log({"Avg Val Loss": avg_val_loss})

# Save the fine-tuned model
base_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/finetuned/'
#'/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/interpolation_preprocess/fine_tuned/'
# Create the directory if it doesn't exist
os.makedirs(base_path, exist_ok=True)
# Save the model
torch.save(model.state_dict(), os.path.join(base_path, f'{name_model}.pt'))

wandb.finish()
