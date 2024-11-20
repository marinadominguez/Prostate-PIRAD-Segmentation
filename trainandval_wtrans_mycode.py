import math
import torch
import wandb
import matplotlib.pyplot as plt
from config import (TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BACKGROUND_AS_CLASS, 
                    TRAIN_CUDA, WANDB_TRAIN_NAME, SAVE_INTERVAL, SAVE_CHECKPOINT_PATH, 
                    BCE_WEIGHTS, DICE_WEIGHT)
from torch.nn import CrossEntropyLoss
#from dataset import get_train_val_test_Dataloaders
from dataset_dicom import get_train_val_test_Dataloaders
from utils import resize_image_and_label
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from unet3d import UNet3D
from transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import BinaryFocalLoss, DiceLoss, DiceCELoss, DiceWCELoss, JaccardCELoss, JaccardWCELoss, JaccardMulticlassLoss, TverskyLoss
import os
import json
from my_focal_loss import CustomFocalLoss

# Set environment variable for allocator max split size
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# select the device to be used for training: CUDA_VISIBLE_DEVICES=1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Initialize W&B
wandb.init(project="3d-unet-multiclass", entity="marina-dominguez", name=WANDB_TRAIN_NAME)
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
    criterion = CustomFocalLoss(alpha=0.25, gamma=2.0, reduction='mean').cuda()
    
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
    criterion = CustomFocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
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
                
        image = data['image'].to(device) #torch.Size([1, 1, 20, 256, 256])
        ground_truth = data['label'].to(device) # torch.Size([1, 1, 20, 256, 256])

        image, ground_truth = resize_image_and_label(image, ground_truth)

        image = image.float() # torch.Size([1, 1, 32, 320, 320])

        ground_truth = ground_truth.long() # torch.Size([1, 1, 32, 320, 320]) = [batch_size, 1, depth, height, width]

        #image = image.half()
        #model = model.half()  # Convert model to half precision
        
        optimizer.zero_grad()

        output = model(image)
        # make the labels occupy less memory
        del image

        invalid_labels = (ground_truth < 0) | (ground_truth >= NUM_CLASSES)
        if torch.any(invalid_labels):
            print("Invalid labels found:", ground_truth[invalid_labels])
        
        # Visualization of the output of the model after softmax, the predicted label map for each class (in each channel)
        if epoch % 10 == 0 and i == 0:  
            if output.max() > 1 and output.min() >= 0:
                softmax_output = F.softmax(output, dim=1)
                print(f"Applied the softmax for the output of image {i} in epoch {epoch}")
            else:
                print(f"Output of image {i} in epoch {epoch} is already in the form of probabilities")
                softmax_output = output
            fig, axs = plt.subplots(1, softmax_output.shape[1] + 1, figsize=(15, 5), gridspec_kw={'width_ratios': [1] * softmax_output.shape[1] + [0.05]})
            # `gridspec_kw={'width_ratios': [1] * softmax_output.shape[1] + [0.05]}` ajusta el ancho de las subgráficas para dejar espacio para la barra de color
            for j in range(softmax_output.shape[1]):
                image = softmax_output[0, j, 0, :, :].detach().cpu().numpy()
                img = axs[j].imshow(image, cmap='gray')
                axs[j].set_title(f"Channel {j+1}")
                axs[j].axis('off')
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
        # apply sigmoid to the output of the model
        output = F.sigmoid(output)
        print(f"Output shape: {output.shape}")
        
        # get same shape for the ground truth as the output of the model
        ground_truth = ground_truth.squeeze(1) # torch.Size([1, 32, 320, 320])
        one_hot_gt = F.one_hot(ground_truth, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        print(f"Ground truth shape: {one_hot_gt.shape}")    

        loss = criterion(output, one_hot_gt)

        print('Loss: ', loss)
        # wandb.log({"Loss": loss})
        loss.backward()

        if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item()
        torch.cuda.empty_cache()
        # torch.cuda.set_per_process_memory_fraction(0.5, device=2)  # Only use 50% of GPU memory

    # Calculate the average training loss for the current epoch
    avg_train_loss = train_loss / len(train_dataloader)

    # Log the average training loss to W&B
    wandb.log({"Avg Train Loss": avg_train_loss})

    ## -------------------------------------- Validation phase ---------------------------------------------
    model.eval()  # set the model to evaluation mode
    print("Validation...")
    with torch.no_grad():
        val_loss = 0.0
        for data in val_dataloader:  # iterate over validation data
            image = data['image'].to(device) #torch.Size([1, 1, 20, 256, 256])
            ground_truth = data['label'].to(device) # torch.Size([1, 1, 20, 256, 256])

            image, ground_truth = resize_image_and_label(image, ground_truth) 
            
            image = image.float() # torch.Size([1, 1, 32, 320, 320])

            ground_truth = ground_truth.long()
            # torch.Size([1, 1, 32, 320, 320])

            # image = image.half()
            # model = model.half()  # Convert model to half precision

            output = model(image)
            # Visualization of the output of the model after softmax, the predicted label map for each class (in each channel)
            if epoch % 10 == 0:  
                if output.max() > 1 and output.min() >= 0:
                    softmax_output = F.softmax(output, dim=1)
                    print(f"Applied the softmax for the output of image {i} in epoch {epoch}")
                else:
                    print(f"Output of image {i} in epoch {epoch} is already in the form of probabilities")
                    softmax_output = output
                fig, axs = plt.subplots(1, softmax_output.shape[1] + 1, figsize=(25, 5), gridspec_kw={'width_ratios': [1] * softmax_output.shape[1] + [0.05]})
                # `gridspec_kw={'width_ratios': [1] * softmax_output.shape[1] + [0.05]}` ajusta el ancho de las subgráficas para dejar espacio para la barra de color
                
                for j in range(softmax_output.shape[1]):
                    image = softmax_output[0, j, 0, :, :].detach().cpu().numpy()
                    img = axs[j].imshow(image, cmap='gray')
                    axs[j].set_title(f"Channel {j+1}")
                    axs[j].axis('off')
                
                # Crear el eje para la barra de color al final
                fig.colorbar(img, cax=axs[-1], orientation='vertical')
                axs[-1].set_title('Pixel\nRange')
                plt.tight_layout()
                
                # Guardar la trama en wandb
                wandb.log({"ValidationSet Predicted Label Map": [wandb.Image(plt, caption="Predicted Label Map")]})
                print("Visualization of the predicted label maps in the validation set for each channel in wandb")
                
                # Cerrar la trama
                plt.close()
            #####################################################################################################################################
            # apply sigmoid to the output of the model
            output = F.sigmoid(output)
            print(f"Output shape: {output.shape}")
            
            # get same shape for the ground truth as the output of the model
            ground_truth = ground_truth.squeeze(1) # torch.Size([1, 32, 320, 320])
            one_hot_gt = F.one_hot(ground_truth, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            print(f"Ground truth shape: {one_hot_gt.shape}")   
              
            loss = criterion(output, one_hot_gt)
            print('Validation Loss: ', loss)
            # wandb.log({"Eval Loss": loss.item()})
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    wandb.log({"Avg Val Loss": avg_val_loss})

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
