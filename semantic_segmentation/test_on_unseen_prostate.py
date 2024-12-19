import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
import pydicom
from ..preprocessing.dataset_hosp import Prostate3DDataset, get_train_val_test_Dataloaders
import ..config.config as cfg

# Assuming your model is defined in a file called unet_model.py
from ..models.unet3d import UNet3D


def load_model(model_path, device):
    model = UNet3D(in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES)  # Adjust in_channels and num_classes as necessary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            
            outputs = model(images.unsqueeze(1))
            preds = (outputs > 0.5).float()
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    return all_labels, all_preds


def calculate_metrics(labels, preds):
    labels_flat = labels.flatten()
    preds_flat = preds.flatten()
    
    accuracy = accuracy_score(labels_flat, preds_flat)
    f1 = f1_score(labels_flat, preds_flat, average='binary')
    
    return accuracy, f1

def main():
    # Paths
    json_file = cfg.DATASET_PATH
    model_path = cfg.CHECKPOINT_PATH
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset
    _, _, test_loader = get_train_val_test_Dataloaders(json_file, batch_size=1, val_split=0.1, test_split=0.8, random_seed=42)

    # Load pretrained model
    model = load_model(model_path, device)

    # Evaluate model
    labels, preds = evaluate_model(model, test_loader, device)

    # Calculate metrics
    accuracy, f1 = calculate_metrics(labels, preds)

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')

if __name__ == "__main__":
    main()
