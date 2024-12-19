import sys
import os

# Add the project directory (.) to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import torch
import matplotlib.pyplot as plt
from ..config.config import (TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS,  
                    TRAIN_CUDA, SAVE_INTERVAL, SAVE_CHECKPOINT_PATH, BACKGROUND_AS_CLASS, 
                    GAMMA, ALPHA) #, FOREGROUND_WEIGHT, DICE_WEIGHT)
                    #BCE_WEIGHTS, DICE_WEIGHT)
# from torch.nn import CrossEntropyLoss
#from dataset import get_train_val_test_Dataloaders
from ..preprocessing.dataset_hosp import get_train_val_test_Dataloaders
from ..utils.utils import resize_image_and_label
from torch.optim import Adam
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from ..models.unet3d import UNet3D
from ..training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from loss import BinaryFocalLoss, DiceLoss, DiceCELoss, DiceWCELoss, JaccardCELoss, JaccardWCELoss, JaccardMulticlassLoss, TverskyLoss
from ..training.my_focal_loss import CustomFocalLoss, CustomFocalForegroundDiceLoss
from ..config.config import config
#import segmentation_models_pytorch_3d as smp
#from segmentation_models_pytorch.losses import FocalLoss, DiceLoss, JaccardLoss, TverskyLoss
#from torch import nn

def dice_coef(pred, target, smooth=1e-5):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# Set environment variable for allocator max split size
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# select the device to be used for training: CUDA_VISIBLE_DEVICES=1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

# Initialize model
model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES) 
train_transforms = train_transform
val_transforms = val_transform

# Initialize DataLoaders, transformations & criterion (depending on whether cuda is available)
if torch.cuda.is_available() and TRAIN_CUDA:
    device = torch.device('cuda')
    model = model.to(device)
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
    # criterion = DiceWCELoss(weight=torch.Tensor(BCE_WEIGHTS).cuda())
    
    # Usage in your training script
    #criterion = CustomFocalLoss(alpha=0.25, gamma=2.0, reduction='mean').cuda()
    criterion = CustomFocalForegroundDiceLoss(alpha=ALPHA, gamma=GAMMA, reduction='mean').cuda()#, dice_weight=DICE_WEIGHT, foreground_weight=FOREGROUND_WEIGHT).cuda()

    print('cuda available! Training initialized on gpu ...')

elif not torch.cuda.is_available() and TRAIN_CUDA:
    criterion = CustomFocalForegroundDiceLoss(alpha=ALPHA, gamma=GAMMA, reduction='mean') #, dice_weight=DICE_WEIGHT, foreground_weight=FOREGROUND_WEIGHT)
    train_transforms = train_transform
    val_transforms = val_transform
    print('cuda not available! Training initialized on cpu ...')

train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, 
                                                                     val_transforms=val_transforms, 
                                                                     test_transforms= val_transforms)

# Initialize optimizer
optimizer = Adam(params=model.parameters())

# Learning Rate Scheduler
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3, verbose=True) # probar sin scheduler, ya que en el otro proyecto hacia que explotara la loss, no cambia nada

# Early Stopping parameters
early_stopping_patience = 10
epochs_since_improvement = 0

# Best model (until a given epoch) parameters
best_val_loss = math.inf # Initialize minimum loss to a high value and best model path to None
best_model_path = ""
save_interval = SAVE_INTERVAL

# Training loop
for epoch in range(TRAINING_EPOCH):
    print(f"Epoch {epoch}/{TRAINING_EPOCH}")
    train_loss = 0.0 # to track the training loss as the model trains
    train_dice_foreground = 0.0
    train_dice_background = 0.0
    total_foreground_coverage = 0.0
    model.train() # prep model for training
    accumulation_steps = 1
    ## -----------------------------------------TRAINING-------------------------------------------------
    print("Training...")

    torch.cuda.empty_cache()  # Clear memory cache

    for i, data in enumerate(train_dataloader): 
        print(f"Batch {i}/{len(train_dataloader)}")
        
        image = data[0].to(device) #torch.Size([1, 1, 20, 256, 256])
        ground_truth = data[1].to(device) # torch.Size([1, 1, 20, 256, 256])

        image, ground_truth = resize_image_and_label(image, ground_truth)

        foreground_coverage = (ground_truth == 1).float().mean().item()
        total_foreground_coverage += foreground_coverage
        print(f"Foreground Coverage: {foreground_coverage}")

        image = image.float() # torch.Size([1, 1, 32, 320, 320])
        
        ground_truth = ground_truth.long() # torch.Size([1, 1, 32, 320, 320]) = [batch_size, 1, depth, height, width]
        print(f"Ground truth shape: {ground_truth.shape}")
        print(f"Unique values in ground truth: {torch.unique(ground_truth)}")
        
        optimizer.zero_grad()

        output = model(image)
        print(f"Output shape: {output.shape}") # torch.Size([1, 2, 32, 320, 320])
        print(f"Range of values in output: {output.min()} - {output.max()}")
        # make the labels occupy less memory
        del image

        invalid_labels = (ground_truth < 0) | (ground_truth >= NUM_CLASSES)
        if torch.any(invalid_labels):
            print("Invalid labels found:", ground_truth[invalid_labels])
            # if it goes inside, it means that there are invalid labels in the ground truth
            raise ValueError("Invalid labels found in the ground truth")
        
        # Visualization of the output of the model after softmax, the predicted label map for each class (in each channel)
        if i == 0 and epoch % 20 == 0: # visualize the first image every 20 epochs
            if output.max() > 1 and output.min() <= 0:
                sigmoid_output = F.sigmoid(output)
                print(f"Applied the softmax for the output of image {i} in epoch {epoch}")
                print(f"sigmoid_output shape: {sigmoid_output.shape}")
            else:
                print(f"Output of image {i} in epoch {epoch} is already in the form of probabilities")
                sigmoid_output = output
                print(f"sigmoid_output shape: {sigmoid_output.shape}")
            
            fig, axs = plt.subplots(1, sigmoid_output.shape[1] + 1, figsize=(15, 5), gridspec_kw={'width_ratios': [1] * sigmoid_output.shape[1] + [0.05]})
            for j in range(sigmoid_output.shape[1]):
                image = sigmoid_output[0, j, 0, :, :].detach().cpu().numpy()
                img = axs[j].imshow(image, cmap='gray')
                axs[j].set_title(f"Sigmoid output first layer, channel {j}")
                axs[j].axis('off')
            # Crear el eje para la barra de color al final
            fig.colorbar(img, cax=axs[-1], orientation='vertical')
            axs[-1].set_title('Pixel\nRange')
            plt.tight_layout()
            # Guarda en output/training_outputs
            plt.savefig(f"output/training_outputs/epoch_{epoch}_batch_{i}_output.png")
            print("Visualization of the predicted label maps saved in output/training_outputs")
            plt.close()
        #####################################################################################################################################
        # apply sigmoid to the output of the model
        output = F.sigmoid(output)
        print(f"Output shape after sigmoid: {output.shape}")
        print(f"Range of values in output after sigmoid: {output.min()} - {output.max()}")
        
        # get same shape for the ground truth as the output of the model
        ground_truth = ground_truth.squeeze(1) # torch.Size([1, 32, 320, 320])
        one_hot_gt = F.one_hot(ground_truth, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        print(f"Ground truth shape: {one_hot_gt.shape}")
        
        loss = criterion(output, one_hot_gt)

        print('Loss: ', loss)

        # Separate foreground and background Dice
        background_mask = (ground_truth == 0).float()
        foreground_mask = (ground_truth == 1).float()

        dice_background = dice_coef(output[:, 0, :, :, :], background_mask)
        dice_foreground = dice_coef(output[:, 1, :, :, :], foreground_mask)


        print(f"Dice Foreground: {dice_foreground}")
        print(f"Dice Background: {dice_background}")

        loss.backward()

        if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item()
        train_dice_foreground += dice_foreground.item()
        train_dice_background += dice_background.item()

        torch.cuda.empty_cache()
        # torch.cuda.set_per_process_memory_fraction(0.5, device=2)  # Only use 50% of GPU memory

    # Calculate the average training loss for the current epoch
    avg_train_loss = train_loss / len(train_dataloader)

    print(f"Average training loss: {avg_train_loss}")

    # calculate the average dice coefficient for the current epoch
    avg_dice_foreground = train_dice_foreground / len(train_dataloader)
    avg_dice_background = train_dice_background / len(train_dataloader)

    print(f"Average Dice Foreground: {avg_dice_foreground}")
    print(f"Average Dice Background: {avg_dice_background}")

    # Log the average foreground coverage to W&B
    avg_foreground_coverage = total_foreground_coverage / len(train_dataloader)
    print(f"Average Foreground Coverage: {avg_foreground_coverage}")

    ## -------------------------------------- Validation phase ---------------------------------------------
    model.eval()  # set the model to evaluation mode
    print("Validation...")
    with torch.no_grad():
        val_loss = 0.0
        val_dice_foreground = 0.0
        val_dice_background = 0.0
        for data in val_dataloader: 
            image = data[0].to(device)
            ground_truth = data[1].to(device)

            image, ground_truth = resize_image_and_label(image, ground_truth) 
            
            image = image.float()

            ground_truth = ground_truth.long()

            output = model(image)
            print(f"The values in the output of the model are in the range: {output.min()} - {output.max()}")

            # Visualization of the output of the model after softmax, the predicted label map for each class (in each channel)
            if epoch % 20 == 0 and i == 0: # visualize only the first image of the first batch
                if output.max() > 1 and output.min() <= 0:
                    sigmoid_output = F.sigmoid(output)
                    print(f"Applied the softmax for the output of image {i} in epoch {epoch}")
                    print(f"sigmoid_output shape: {sigmoid_output.shape}")
                else:
                    print(f"Output of image {i} in epoch {epoch} is already in the form of probabilities")
                    sigmoid_output = output
                    print(f"sigmoid_output shape: {sigmoid_output.shape}")
                fig, axs = plt.subplots(1, sigmoid_output.shape[1] + 1, figsize=(25, 5), gridspec_kw={'width_ratios': [1] * sigmoid_output.shape[1] + [0.05]})
                for j in range(sigmoid_output.shape[1]): 
                    image = sigmoid_output[0, j, 0, :, :].detach().cpu().numpy()
                    img = axs[j].imshow(image, cmap='gray')
                    axs[j].set_title(f"Sigmoid output first layer, channel {j}")
                    axs[j].axis('off')
                # Crear el eje para la barra de color al final
                fig.colorbar(img, cax=axs[-1], orientation='vertical')
                axs[-1].set_title('Pixel\nRange')
                plt.tight_layout()
                
                # Guarda en output/validation_outputs
                plt.savefig(f"output/validation_outputs/epoch_{epoch}_batch_{i}_output.png")
                print("Visualization of the predicted label maps in the validation set for each channel in output/validation_outputs")
                
                # Cerrar la trama
                plt.close()
            #####################################################################################################################################
            # apply sigmoid to the output of the model
            output = F.sigmoid(output)
            print(f"Output shape: {output.shape}")
            print(f"Range of values in output after sigmoid: {output.min()} - {output.max()}")
            
            print("get same shape for the ground truth as the output of the model")
            ground_truth = ground_truth.squeeze(1) 
            one_hot_gt = F.one_hot(ground_truth, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            print(f"Ground truth shape: {one_hot_gt.shape}")   
              
            loss = criterion(output, one_hot_gt)
            print('Validation Loss: ', loss)

            # Calculate Dice for foreground and background
            background_mask = (ground_truth == 0).float()
            foreground_mask = (ground_truth == 1).float()

            dice_background = dice_coef(output[:, 0, :, :, :], background_mask)
            dice_foreground = dice_coef(output[:, 1, :, :, :], foreground_mask)

            print(f"Dice Foreground: {dice_foreground}")
            print(f"Dice Background: {dice_background}")

            val_loss += loss.item()
            val_dice_foreground += dice_foreground.item()
            val_dice_background += dice_background.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Average validation loss: {avg_val_loss}")

    avg_val_dice_foreground = val_dice_foreground / len(val_dataloader)
    avg_val_dice_background = val_dice_background / len(val_dataloader)
    print(f"Average validation dice foreground: {avg_val_dice_foreground}")
    print(f"Average validation dice background: {avg_val_dice_background}")

    ## ------------------------------------- Checkpointing ------------------------------------------------
    # Actualización de la lógica de guardado
    if epoch % save_interval == 0 or avg_val_loss < best_val_loss:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss}")

        best_model_path = f'{SAVE_CHECKPOINT_PATH}/best_model_epoch{epoch}.pt'
        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved to {best_model_path} at epoch {epoch}")
        epochs_since_improvement = 0  # resetear contador
    else: # nunca se cumple el else porque siempre se guarda el modelo cada 10 épocas y se resetea el contador
        epochs_since_improvement += 1
        print(f"No improvement in validation loss for {epochs_since_improvement} epochs")
    
    # Early Stopping
    if epochs_since_improvement == early_stopping_patience:
        print(f'Early stopping triggered after {epoch} epochs!')
        break
    if avg_val_loss - best_val_loss < 0.00001 and epochs_since_improvement > early_stopping_patience-1:
        # tampoco entra nunca porque se resetea el contador cada 10 épocas
        print(f'Early stopping triggered after {epoch} epochs!')
        break