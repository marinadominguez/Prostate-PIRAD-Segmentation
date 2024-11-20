import sys
import os

# Add the project directory (HOSP_pipeline) to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import torch
import wandb
import matplotlib.pyplot as plt
from HOSP_pipeline.config.config import (TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS,  
                    TRAIN_CUDA, WANDB_TRAIN_NAME, SAVE_INTERVAL, SAVE_CHECKPOINT_PATH)
                    #BCE_WEIGHTS, DICE_WEIGHT)
# from torch.nn import CrossEntropyLoss
#from dataset import get_train_val_test_Dataloaders
from HOSP_pipeline.preprocessing.dataset_hosp import get_train_val_test_Dataloaders
from HOSP_pipeline.utils.utils import resize_image_and_label
from torch.optim import Adam
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from HOSP_pipeline.models.unet3d import UNet3D
from HOSP_pipeline.training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from loss import BinaryFocalLoss, DiceLoss, DiceCELoss, DiceWCELoss, JaccardCELoss, JaccardWCELoss, JaccardMulticlassLoss, TverskyLoss
import os
# import json
from HOSP_pipeline.training.my_focal_loss import CustomFocalLoss
from HOSP_pipeline.config.config import config
import segmentation_models_pytorch_3d as smp
import torch.nn.functional as F
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss, JaccardLoss, TverskyLoss
from torch import nn

# Set environment variable for allocator max split size
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# select the device to be used for training: CUDA_VISIBLE_DEVICES=1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Initialize W&B
wandb.init(project="hops_pipeline", entity="marina-dominguez", name=WANDB_TRAIN_NAME)
# Log the entire config dictionary
wandb.config.update(config)
# Now you can access it like this:
print("Dataset path:", wandb.config.dataset_path)
print("Number of epochs:", wandb.config.training_epoch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if BACKGROUND_AS_CLASS: NUM_CLASSES += 1
# Initialize model
# model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES

# Definir el modelo con UNet++ y ResNet como encoder
model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # ResNet-34 como encoder preentrenado
    encoder_weights=None,     # Cargar pesos preentrenados en ImageNet
    in_channels=1,                  # Cambia según el número de canales de tus imágenes (3 si son RGB)
    classes=1                       # Número de clases de salida, para binario usa 1
)

train_transforms = train_transform
val_transforms = val_transform

class StableFocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(StableFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        # Convertir a probabilidades si es necesario
        outputs = torch.clamp(outputs, min=1e-7, max=1 - 1e-7)
        bce_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probabilidades de predicción
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# Define combined loss function
class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.6, weight_factor=1):  
        super().__init__()
        self.dice_loss = DiceLoss(mode='binary')
        self.focal_loss = StableFocalLoss(alpha=weight_factor, gamma=2)
        self.alpha = alpha

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        targets = targets.float()
        
        dice = self.dice_loss(outputs, targets)
        #focal = self.focal_loss(outputs, targets)
        
        print(f"Dice Loss: {dice.item()}")#, Focal Loss: {focal.item()}")
        #combined_loss = self.alpha * dice + (1 - self.alpha) * focal
        #print(f"Combined Loss: {combined_loss.item()}")
        
        return dice

class WeightedDiceLoss(nn.Module):
    def __init__(self, weight_background=0.1, weight_foreground=0.9):
        super(WeightedDiceLoss, self).__init__()
        self.weight_background = weight_background
        self.weight_foreground = weight_foreground

    def forward(self, predictions, targets):
        smooth = 1e-5  # Factor para evitar divisiones por cero
        predictions = torch.sigmoid(predictions)  # Asegúrate de que predictions esté en forma de probabilidad

        # Calcular la intersección y unión para ambas clases
        intersection_background = ((1 - targets) * (1 - predictions)).sum()
        union_background = ((1 - targets) + (1 - predictions)).sum()
        
        intersection_foreground = (targets * predictions).sum()
        union_foreground = (targets + predictions).sum()

        # Dice para fondo y para la clase de interés
        dice_background = (2.0 * intersection_background + smooth) / (union_background + smooth)
        dice_foreground = (2.0 * intersection_foreground + smooth) / (union_foreground + smooth)

        # Ponderación de cada clase
        weighted_dice = (self.weight_background * (1 - dice_background)) + \
                        (self.weight_foreground * (1 - dice_foreground))
        
        return weighted_dice
    
class WeightedCombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, weight_factor=10):  
        super().__init__()
        self.dice_loss = smp.utils.losses.DiceLoss(mode="binary")
        self.focal_tversky_loss = smp.utils.losses.TverskyLoss(mode="binary", alpha=beta, beta=1 - beta, gamma=gamma)
        self.alpha = alpha
        self.weight_factor = weight_factor

    def forward(self, outputs, targets):
        targets = targets.float()
        dice = self.dice_loss(outputs, targets) * self.weight_factor  # Emphasize foreground in Dice
        focal_tversky = self.focal_tversky_loss(outputs, targets)
        return self.alpha * dice + (1 - self.alpha) * focal_tversky

#criterion = WeightedCombinedLoss(alpha=0.7, beta=0.3, gamma=2.0, weight_factor=20).cuda()
    
def dice_coef(pred, target, smooth=1e-5):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# Initialize DataLoaders, transformations & criterion (depending on whether cuda is available)
if torch.cuda.is_available() and TRAIN_CUDA:
    device = torch.device('cuda')
    model = model.to(device)
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
    # criterion = DiceWCELoss(weight=torch.Tensor(BCE_WEIGHTS).cuda())
    
    # Usage in your training script
    #criterion = smp.utils.losses.DiceLoss()
    # Use the combined loss in your training loop
    #criterion = CombinedLoss().cuda()
    criterion = WeightedDiceLoss().cuda()

    # criterion = CustomFocalLoss(alpha=0.25, gamma=2.0, reduction='mean').cuda()
    # criterion = BinaryFocalLoss().cuda()
    # criterion = FocalLoss(gamma=2, alpha=0.25).cuda()
    # criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS).cuda())
    # criterion = DiceLoss().cuda()
    # criterion = JaccardMulticlassLoss().cuda()
    # criterion = TverskyLoss().cuda()
    # criterion = FocalLoss(gamma=2, alpha=0.25).cuda()
    # criterion = JaccardWCELoss(num_classes=NUM_CLASSES).cuda()
    # criterion = DiceCELoss(num_classes=NUM_CLASSES, dice_weight=DICE_WEIGHT).cuda()
    # criterion = JaccardCELoss(num_classes=NUM_CLASSES).cuda()
    print('cuda available! Training initialized on gpu ...')

elif not torch.cuda.is_available() and TRAIN_CUDA:
    # criterion = smp.utils.losses.DiceLoss()
    #criterion = CombinedLoss()
    criterion = WeightedDiceLoss()
    #criterion = CustomFocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    # criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
    # criterion = DiceLoss()
    # criterion = JaccardMulticlassLoss()
    #criterion = DiceWCELoss(weight=torch.Tensor(BCE_WEIGHTS))
    # criterion = TverskyLoss()
    # criterion = FocalLoss(gamma=2, alpha=0.25)
    # criterion = BinaryFocalLoss()

    #criterion = FocalLoss(gamma=0.7)
    # To make it behaves as CrossEntropy loss
    # criterion = FocalLoss(gamma=0)

    # criterion = JaccardWCELoss(num_classes=NUM_CLASSES)
    # criterion = DiceCELoss(num_classes=NUM_CLASSES, dice_weight=DICE_WEIGHT)
    # criterion = JaccardCELoss(num_classes=NUM_CLASSES)
    train_transforms = train_transform
    val_transforms = val_transform
    print('cuda not available! Training initialized on cpu ...')

train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms= train_transforms, 
                                                                     val_transforms=val_transforms, 
                                                                     test_transforms= val_transforms)

# Initialize optimizer
# Reduce el learning rate inicial
learning_rate = 1e-4  # Baja el valor (por ejemplo, 1e-4 o incluso 1e-5)
optimizer = Adam(model.parameters(), lr=learning_rate)

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
    model.train() # prep model for training
    accumulation_steps = 1
    ## -----------------------------------------TRAINING-------------------------------------------------
    print("Training...")
    """
    Transformed image shape: torch.Size([1, 20, 256, 256])
    Transformed label shape: torch.Size([1, 20, 256, 256])
    image shape:         torch.Size([12, 1, 20, 256, 256])
    ground truth shape:  torch.Size([12, 1, 20, 256, 256])
    Resized image shape: torch.Size([12, 1, 32, 320, 320])
    Resized label shape: torch.Size([12, 1, 32, 320, 320])
    """

    torch.cuda.empty_cache()  # Clear memory cache

    for i, data in enumerate(train_dataloader): 
        print(f"Batch {i}/{len(train_dataloader)}")
                
        images = data[0].to(device)
        labels = data[1].to(device)

        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        
        image, ground_truth = resize_image_and_label(images, labels)
        foreground_coverage = (ground_truth == 1).float().mean().item()
        foreground_coverage += foreground_coverage
        print(f"Foreground Coverage: {foreground_coverage}")
        #wandb.log({"Foreground Coverage": foreground_coverage})
        image = image.float()
        ground_truth = ground_truth.long()

        # verificamos que ground truth este en modo binario
        print(f"Ground truth unique values: {torch.unique(ground_truth)}")

        optimizer.zero_grad()

        output = model(image)
        print(f"Output shape: {output.shape}")
        # make the labels occupy less memory
        del image

        # Check for invalid labels before one-hot encoding
        invalid_labels = (ground_truth < 0) | (ground_truth >= NUM_CLASSES)
        if torch.any(invalid_labels):
            print("Invalid labels found, correcting values")
            ground_truth = torch.clamp(ground_truth, min=0, max=NUM_CLASSES - 1)
        
        print(f"Shape of model output before softmax: {output.shape}")
        # Visualization of the output of the model after softmax, the predicted label map for each class (in each channel)
        if epoch % 10 == 0 and i == 0:  
            if output.max() > 1 and output.min() >= 0:
                # softmax_output = F.softmax(output, dim=1)
                sigmoid_output = F.sigmoid(output)
                print(f"Applied the softmax for the output of image {i} in epoch {epoch}")
            else:
                print(f"Output of image {i} in epoch {epoch} is already in the form of probabilities")
                #softmax_output = output
                sigmoid_output = output
            
            fig, axs = plt.subplots(1, sigmoid_output.shape[1] + 1, figsize=(15, 5), gridspec_kw={'width_ratios': [1] * sigmoid_output.shape[1] + [0.05]})
            # `gridspec_kw={'width_ratios': [1] * softmax_output.shape[1] + [0.05]}` ajusta el ancho de las subgráficas para dejar espacio para la barra de color
            #for j in range(sigmoid_output.shape[1]):
            image = sigmoid_output[0, 0, 0, :, :].detach().cpu().numpy()
            img = axs[0].imshow(image, cmap='gray')
            axs[0].set_title(f"Sigmoid Output first slice")
            axs[0].axis('off')
            # Crear el eje para la barra de color al final
            fig.colorbar(img, cax=axs[-1], orientation='vertical')
            axs[-1].set_title('Pixel\nRange')
            plt.tight_layout()

            # Guardar la trama en wandb
            wandb.log({"Predicted Label Map": [wandb.Image(plt, caption="Predicted Label Map")]})
            print("Visualization of the predicted label maps for each channel in wandb")
            # Cerrar la trama
            plt.close()
        #####################################################################################################################################
        # Apply sigmoid to output
        output = F.sigmoid(output)
        print(f"Output shape: {output.shape}")
        
        # Ensure ground_truth is compatible with one-hot encoding
        #ground_truth = ground_truth.squeeze(1)
        print(f"Ground truth shape: {ground_truth.shape}")
        
        #one_hot_gt = F.one_hot(ground_truth, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        #print(f"Ground truth shape after one-hot encoding: {one_hot_gt.shape}")   

        loss = criterion(output, ground_truth)

        print('Loss: ', loss)
        
        # Separate foreground and background Dice
        background_mask = (ground_truth == 0).float()
        foreground_mask = (ground_truth == 1).float()

        dice_background = dice_coef(output, background_mask)
        dice_foreground = dice_coef(output, foreground_mask)

        # wandb.log({"Loss": loss})
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

    # Log the average training loss to W&B
    wandb.log({"Avg Train Loss": avg_train_loss})

    # calculate the average dice coefficient for the current epoch
    avg_dice_foreground = train_dice_foreground / len(train_dataloader)
    avg_dice_background = train_dice_background / len(train_dataloader)

    # Log the average dice coefficient to W&B
    wandb.log({"Avg Dice Foreground": avg_dice_foreground})
    wandb.log({"Avg Dice Background": avg_dice_background})

    # Log the average foreground coverage to W&B
    #avg_foreground_coverage = foreground_coverage / len(train_dataloader)
    #wandb.log({"Avg Foreground Coverage": avg_foreground_coverage})

    ## -------------------------------------- Validation phase ---------------------------------------------
    model.eval()  # set the model to evaluation mode
    print("Validation...")
    with torch.no_grad():
        val_loss = 0.0
        dice_foreground_total = 0.0
        dice_background_total = 0.0
        for data in val_dataloader:  # iterate over validation data
            images = data[0].to(device)
            labels = data[1].to(device)

            image, ground_truth = resize_image_and_label(images, labels)
            image = image.float()
            ground_truth = ground_truth.long()

            output = model(image)
            
            # Visualization of the output of the model after softmax, the predicted label map for each class
            if epoch % 10 == 0:  
                sigmoid_output = F.sigmoid(output)
                fig, axs = plt.subplots(1, sigmoid_output.shape[1] + 1, figsize=(25, 5),
                                        gridspec_kw={'width_ratios': [1] * sigmoid_output.shape[1] + [0.05]})
                image = sigmoid_output[0, 0, 0, :, :].detach().cpu().numpy()
                img = axs[0].imshow(image, cmap='gray')
                axs[0].set_title("Sigmoid Output first slice")
                axs[0].axis('off')
                fig.colorbar(img, cax=axs[-1], orientation='vertical')
                axs[-1].set_title('Pixel\nRange')
                plt.tight_layout()
                wandb.log({"ValidationSet Predicted Label Map": [wandb.Image(plt, caption="Predicted Label Map")]})
                plt.close()

            # Apply sigmoid to the output
            output = F.sigmoid(output)
            
            # Ensure ground_truth is compatible with one-hot encoding
            #ground_truth = ground_truth.squeeze(1)  # Remove extra dimension if present

            # Debug: Check range of ground_truth values
            invalid_labels = (ground_truth < 0) | (ground_truth >= NUM_CLASSES)
            if torch.any(invalid_labels):
                print("Warning: Found out-of-bounds values in ground_truth, setting them to zero")
                ground_truth = torch.clamp(ground_truth, min=0, max=NUM_CLASSES - 1)

            # Apply one-hot encoding
            #one_hot_gt = F.one_hot(ground_truth, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            
            loss = criterion(output, ground_truth)
            print('Validation Loss: ', loss)

            # Calculate Dice for foreground and background
            background_mask = (ground_truth == 0).float()
            foreground_mask = (ground_truth == 1).float()

            dice_background = dice_coef(output, background_mask)
            dice_foreground = dice_coef(output, foreground_mask)

            val_loss += loss.item()
            dice_foreground_total += dice_foreground.item()
            dice_background_total += dice_background.item()
        
        # Log average metrics
        avg_val_loss = val_loss / len(val_dataloader)
        avg_dice_foreground = dice_foreground_total / len(val_dataloader)
        avg_dice_background = dice_background_total / len(val_dataloader)

        wandb.log({"Avg Val Loss": avg_val_loss})
        wandb.log({"Avg Dice Foreground Val": avg_dice_foreground})
        wandb.log({"Avg Dice Background Val": avg_dice_background})

    ## ------------------------------------- Checkpointing ------------------------------------------------
    # Actualización de la lógica de guardado
    if epoch % save_interval == 0 or avg_val_loss < best_val_loss:
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss}")

        best_model_path = f'{SAVE_CHECKPOINT_PATH}/best_model_epoch{epoch}.pt'
        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved to {best_model_path} at epoch {epoch}")
        wandb.save(best_model_path)
        epochs_since_improvement = 0  # resetear contador
    else: # nunca se cumple el else porque siempre se guarda el modelo cada 10 épocas y se resetea el contador
        epochs_since_improvement += 1
        print(f"No improvement in validation loss for {epochs_since_improvement} epochs")
    
    # Learning Rate Scheduling
    # scheduler.step(avg_val_loss)

    # Early Stopping
    if epochs_since_improvement == early_stopping_patience:
        print(f'Early stopping triggered after {epoch} epochs!')
        break
    if avg_val_loss - best_val_loss < 0.00001 and epochs_since_improvement > early_stopping_patience-1:
        # aquí lo miso, tampoco entra nunca porque se resetea el contador cada 10 épocas
        print(f'Early stopping triggered after {epoch} epochs!')
        break

wandb.finish()
