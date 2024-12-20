import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
from HOSP_pipeline.preprocessing.dataset_hosp import get_train_val_test_Dataloaders
from HOSP_pipeline.models.unet3d import UNet3D
from HOSP_pipeline.config.config import (TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS,  
                    TRAIN_CUDA, SAVE_INTERVAL, SAVE_CHECKPOINT_PATH, BACKGROUND_AS_CLASS, 
                    GAMMA, ALPHA, CHECKPOINT_PATH) #, FOREGROUND_WEIGHT, DICE_WEIGHT)
                    #BCE_WEIGHTS, DICE_WEIGHT)  
from HOSP_pipeline.utils.utils import resize_image_and_label
from HOSP_pipeline.training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np
import torch.nn.functional as F
from HOSP_pipeline.config.config import config
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from HOSP_pipeline.training.my_focal_loss import CustomFocalForegroundDiceLoss
from torch.optim import Adam
import math
import json
import pydicom

# Set environment variable for allocator max split size
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# select the device to be used for training: CUDA_VISIBLE_DEVICES=1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

PRETRAINED_NUM_CLASSES = 2  # Asegúrate de que coincide con el modelo entrenado
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ruta al archivo JSON con las anotaciones de las lesiones
annotations_path = 'preprocessing/interpolation_preprocess/interpolated_annotated_cancer.json'

# Cargar las anotaciones
with open(annotations_path, 'r') as f:
    annotations = json.load(f)

print(f"Total number of slices: {len(annotations)}")

def dice_coef(pred, target, smooth=1e-5):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def get_slice_idx_from_dicom(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    slice_idx = dicom.InstanceNumber  # el atributo apropiado que indica el número de corte
    print(f"El número de corte de la slice correspondiente: {slice_idx}")
    return slice_idx

def get_largest_non_near_edge_component(binary_mask, border_margin=100): #based on pppm
    """
    Get the largest connected component that doesn't touch the border
    and is not close to the border (within a margin).
    
    Args:
    - binary_mask (np.ndarray): Binary mask where 1 indicates the region of interest.
    - border_margin (int): Margin to exclude from the borders (default: 20 pixels).
    
    Returns:
    - np.ndarray: Binary mask with only the largest connected component in the safe zone.
    """
    # Step 1: Remove connected components that touch the border
    cleared_mask = clear_border(binary_mask)
    
    # Step 2: Define the central "safe zone"
    height, width = cleared_mask.shape
    safe_zone = np.zeros_like(cleared_mask)
    
    # Create a central region (excluding 'border_margin' pixels from each side)
    safe_zone[border_margin:height-border_margin, border_margin:width-border_margin] = 1
    
    # Step 3: Apply the "safe zone" mask to remove components near the borders
    cleared_mask = cleared_mask * safe_zone

    # Step 4: Label remaining connected components
    labeled_mask = label(cleared_mask)

    # Step 5: Get the largest connected component
    if labeled_mask.max() == 0:
        print("No connected components found after clearing borders and applying the safe zone.")
        return np.zeros_like(binary_mask)

    # Step 6: Measure properties of labeled regions    
    regions = regionprops(labeled_mask)
    print(f"Number of regions: {len(regions)}")
    
    if not regions:
        return binary_mask  # No regions left after clearing borders

    # 7: Find the largest region by area
    largest_region = max(regions, key=lambda r: r.area)
    print(f"Largest region area: {largest_region.area}")
    
    # 8: Create a new binary mask with only the largest component
    largest_component_mask = (labeled_mask == largest_region.label).astype(int)  # Use `int` instead of `np.int`
    print(f"largest_component_mask shape: {largest_component_mask.shape}")
    
    return largest_component_mask

def extract_prostate_region(mask):
    """
    Extracts the bounding box of the prostate from a binary mask.
    Returns the coordinates of the bounding box.
    """
    prostate_coords = np.where(mask > 0)
    if prostate_coords[0].size == 0:  # No prostate in this slice
        return None  # or return empty bounds for flexibility

    min_y, max_y = prostate_coords[0].min(), prostate_coords[0].max()
    min_x, max_x = prostate_coords[1].min(), prostate_coords[1].max()
    return min_y, max_y, min_x, max_x

def extract_prostate_region_3d(mask):
    """
    Extracts the bounding box of the prostate from a 3D binary mask.
    Returns the coordinates of the bounding box in z, y, x dimensions.
    """
    prostate_coords = np.where(mask > 0)
    if prostate_coords[0].size == 0:  # No prostate found in the volume
        return None

    min_z, max_z = prostate_coords[0].min(), prostate_coords[0].max()
    min_y, max_y = prostate_coords[1].min(), prostate_coords[1].max()
    min_x, max_x = prostate_coords[2].min(), prostate_coords[2].max()

    print(f"Prostate bounding box: z=[{min_z}, {max_z}], y=[{min_y}, {max_y}], x=[{min_x}, {max_x}]")

    return min_z, max_z, min_y, max_y, min_x, max_x

def pad_to_multiple(tensor, multiple=16):
    depth, height, width = tensor.shape[-3:]
    pad_depth = (multiple - depth % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    padding = (0, pad_width, 0, pad_height, 0, pad_depth)  # Padding in the order (W, H, D)
    padded_tensor = F.pad(tensor, padding)
    return padded_tensor

# Cargar el modelo
pretrained_prostate_model = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
pretrained_prostate_model.load_state_dict(checkpoint)
pretrained_prostate_model = pretrained_prostate_model.to(device)
pretrained_prostate_model.eval()

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

# Initialize new model
model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES) 
train_transforms = train_transform
val_transforms = val_transform

# Inicializar DataLoader
if torch.cuda.is_available() and TRAIN_CUDA:
    device = torch.device('cuda')
    model = model.to(device)
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda 
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

# Early Stopping parameters
early_stopping_patience = 10
epochs_since_improvement = 0

# Best model (until a given epoch) parameters
best_val_loss = math.inf # Initialize minimum loss to a high value and best model path to None
best_model_path = ""
save_interval = SAVE_INTERVAL

# Directorios para guardar las imágenes y etiquetas recortadas
#cropped_image_dir = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/cropped_imgs/images' 
#cropped_label_dir = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/cropped_imgs/labels'  
#os.makedirs(cropped_image_dir, exist_ok=True)
#os.makedirs(cropped_label_dir, exist_ok=True)

# Inicialización
all_ground_truths = []
all_predictions = []
all_images_gt = []
count_visualize_slice_predictions = 0
count_visualize_overlap = 0
iou_scores = []
threshold_selected = 0.202

# Lista para almacenar nuevas anotaciones
new_annotations = []

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
        print(f"Batch {i+1} out of {len(train_dataloader)} batches.")

        # Desempaquetar los datos
        image = data[0].to(device)
        ground_truth = data[1].to(device)

        # Generar un folder_id único basado en el índice del paciente
        #folder_id = f'patient_{i}'

        resized_images, resized_labels = resize_image_and_label(image, ground_truth)
        resized_images = resized_images.float()
        resized_labels = resized_labels.long()

        foreground_coverage = (ground_truth == 1).float().mean().item()
        total_foreground_coverage += foreground_coverage
        print(f"Foreground Coverage: {foreground_coverage}")

        outputs = pretrained_prostate_model(resized_images)
        outputs_sigmoid = torch.sigmoid(outputs)
        prostate_probabilities = outputs_sigmoid[0, 1, :, :, :]  # Shape: (num_slices, H, W)

        # Binary predictions over the volume
        predictions = (prostate_probabilities >= threshold_selected).int()  # Shape: (num_slices, H, W)

        # Apply post-processing over each slice
        for slice_idx in range(predictions.shape[0]):
            predictions[slice_idx, :, :] = torch.tensor(
                get_largest_non_near_edge_component(predictions[slice_idx, :, :].cpu().numpy())
            ).to(device)

        # Now, we have a 3D prostate mask
        prostate_mask = predictions.cpu().numpy()  # Shape: (num_slices, H, W)
        print(f"Prostate mask shape: {prostate_mask.shape}")

        # Find the bounding box over the 3D mask
        bbox = extract_prostate_region_3d(prostate_mask)

        if bbox is not None:
            min_z, max_z, min_y, max_y, min_x, max_x = bbox

            # Apply margins
            margin_z = 0  # Adjust if you want to include more slices
            margin_xy = 10  # Adjust as needed

            min_z = max(min_z - margin_z, 0)
            max_z = min(max_z + margin_z, prostate_mask.shape[0] - 1)

            min_y = max(min_y - margin_xy, 0)
            max_y = min(max_y + margin_xy, prostate_mask.shape[1] - 1)

            min_x = max(min_x - margin_xy, 0)
            max_x = min(max_x + margin_xy, prostate_mask.shape[2] - 1)

            # Crop the images and lesion masks using the bounding box
            image_cropped = resized_images[:, :, min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
            lesion_mask_cropped = resized_labels[:, :, min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
            print(f"Image cropped shape: {image_cropped.shape}") 
            print(f"Image cropped shape: {image_cropped.shape}") # (num_slices, H, W)
            prostate_mask_cropped = prostate_mask[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
            print(f"Prostate mask cropped shape: {prostate_mask_cropped.shape}")

            # save the number of annotated slices for visualization before padding
            num_cropped_slices = image_cropped.shape[2]
            print(f"Number of cropped slices: {num_cropped_slices}")

            # Pad to multiple of 16
            image_cropped = pad_to_multiple(image_cropped, multiple=16)
            lesion_mask_cropped = pad_to_multiple(lesion_mask_cropped, multiple=16)

            print(f"Image cropped shape after padding: {image_cropped.shape}")
            print(f"Lesion mask cropped shape after padding: {lesion_mask_cropped.shape}")

            # Save the cropped images and masks per slice in wandb
            for idx in range(num_cropped_slices):
                slice_num = min_z + idx  # Original slice number

                cropped_image_slice = image_cropped[0, 0, idx, :, :].cpu().numpy()
                cropped_label_slice = lesion_mask_cropped[0, 0, idx, :, :].cpu().numpy()
                print(f"Slice {slice_num} - Image shape: {cropped_image_slice.shape}, Label shape: {cropped_label_slice.shape}")

                # Visualización opcional
                if count_visualize_overlap < 50:
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(cropped_image_slice, cmap='gray')
                    ax[0].axis('off')
                    ax[0].set_title(f'Slice {slice_num}')

                    ax[1].imshow(prostate_mask_cropped[idx, :, :], cmap='gray')
                    ax[1].axis('off')
                    ax[1].set_title(f'Prostate Mask {slice_num}')

                    ax[2].imshow(cropped_label_slice, cmap='gray')
                    ax[2].axis('off')
                    ax[2].set_title(f'Lesion Mask {slice_num}')

                    plt.tight_layout()
                    # Guardar la visualización en wandb
                    wandb.log({"Cropped Images and Masks": [wandb.Image(fig)]})
                    #plt.savefig(f'/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/cropped_imgs/visualization/visualization_patient{i}_slice{slice_num}.png')
                    plt.close()

                    count_visualize_overlap += 1

        else:
            print(f"No prostate found for patient {i}")
            continue  # Pasamos al siguiente paciente
        
        optimizer.zero_grad()
        output = model(image_cropped) 
        print(f"Output shape: {output.shape}")
        print(f"Range of values in output: {output.min()} - {output.max()}")
        # make the labels occupy less memory
        del image_cropped

        invalid_labels = (lesion_mask_cropped < 0) | (lesion_mask_cropped >= NUM_CLASSES)
        if torch.any(invalid_labels):
            print("Invalid labels found:", lesion_mask_cropped[invalid_labels])
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
            # Guardar la trama en wandb
            wandb.log({"Predicted Label Map": [wandb.Image(plt, caption="Predicted Label Map")]})
            print("Visualization of the predicted label maps in wandb")
            # Cerrar la trama
            plt.close()
        #####################################################################################################################################
        # apply sigmoid to the output of the model
        output = F.sigmoid(output)
        print(f"Output shape after sigmoid: {output.shape}")
        print(f"Range of values in output after sigmoid: {output.min()} - {output.max()}")
        
        # get same shape for the ground truth as the output of the model
        lesion_mask_cropped = lesion_mask_cropped.squeeze(1) # torch.Size([1, 32, 320, 320])
        one_hot_gt = F.one_hot(lesion_mask_cropped, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        print(f"Ground truth shape: {one_hot_gt.shape}")
        
        loss = criterion(output, one_hot_gt)

        print('Loss: ', loss)
        # wandb.log({"Loss": loss})

        # Separate foreground and background Dice
        background_mask = (lesion_mask_cropped == 0).float()
        foreground_mask = (lesion_mask_cropped == 1).float()

        dice_background = dice_coef(output, background_mask)
        dice_foreground = dice_coef(output, foreground_mask)

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

    # Log the average training loss to W&B
    wandb.log({"Avg Train Loss": avg_train_loss})
    print(f"Average training loss: {avg_train_loss}")

    # calculate the average dice coefficient for the current epoch
    avg_dice_foreground = train_dice_foreground / len(train_dataloader)
    avg_dice_background = train_dice_background / len(train_dataloader)

    # Log the average dice coefficient to W&B
    wandb.log({"Avg Dice Foreground": avg_dice_foreground})
    print(f"Average Dice Foreground: {avg_dice_foreground}")
    wandb.log({"Avg Dice Background": avg_dice_background})
    print(f"Average Dice Background: {avg_dice_background}")

    # Log the average foreground coverage to W&B
    avg_foreground_coverage = total_foreground_coverage / len(train_dataloader)
    wandb.log({"Avg Foreground Coverage": avg_foreground_coverage})
    print(f"Average Foreground Coverage: {avg_foreground_coverage}")

    ## -------------------------------------- Validation phase ---------------------------------------------
    model.eval()  # set the model to evaluation mode
    print("Validation...")
    with torch.no_grad():
        val_loss = 0.0
        val_dice_foreground = 0.0
        val_dice_background = 0.0
        for data in val_dataloader:  # iterate over validation data
            image = data[0].to(device) #torch.Size([1, 1, 20, 256, 256])
            ground_truth = data[1].to(device) # torch.Size([1, 1, 20, 256, 256])

            resize_image, resize_ground_truth = resize_image_and_label(image, ground_truth) 
            resize_image = resize_image.float() # torch.Size([1, 1, 32, 320, 320])
            resize_ground_truth = resize_ground_truth.long()

            outputs = pretrained_prostate_model(resized_images)
            print(f"The values in the output of the model are in the range: {outputs.min()} - {outputs.max()}")
            outputs_sigmoid = torch.sigmoid(outputs)
            prostate_probabilities = outputs_sigmoid[0, 1, :, :, :]  # Shape: (num_slices, H, W)

            # Binary predictions over the volume
            predictions = (prostate_probabilities >= threshold_selected).int()  # Shape: (num_slices, H, W)

            # Apply post-processing over each slice
            for slice_idx in range(predictions.shape[0]):
                predictions[slice_idx, :, :] = torch.tensor(
                    get_largest_non_near_edge_component(predictions[slice_idx, :, :].cpu().numpy())
                ).to(device)

            # Now, we have a 3D prostate mask
            prostate_mask = predictions.cpu().numpy()  # Shape: (num_slices, H, W)

            # Find the bounding box over the 3D mask
            bbox = extract_prostate_region_3d(prostate_mask)

            if bbox is not None:
                min_z, max_z, min_y, max_y, min_x, max_x = bbox

                # Apply margins
                margin_z = 0  # Adjust if you want to include more slices
                margin_xy = 10  # Adjust as needed

                min_z = max(min_z - margin_z, 0)
                max_z = min(max_z + margin_z, prostate_mask.shape[0] - 1)

                min_y = max(min_y - margin_xy, 0)
                max_y = min(max_y + margin_xy, prostate_mask.shape[1] - 1)

                min_x = max(min_x - margin_xy, 0)
                max_x = min(max_x + margin_xy, prostate_mask.shape[2] - 1)

                # Crop the images and lesion masks using the bounding box
                image_cropped = resized_images[:, :, min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
                print(f"Image cropped shape: {image_cropped.shape}") # 
                lesion_mask_cropped = resized_labels[:, :, min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
                print(f"Image cropped shape: {image_cropped.shape}") # (num_slices, H, W)
                prostate_mask_cropped = prostate_mask[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
                
                # save the number of annotated slices for visualization before padding
                num_cropped_slices = image_cropped.shape[2]

                # Pad to multiple of 16
                image_cropped = pad_to_multiple(image_cropped, multiple=16)
                lesion_mask_cropped = pad_to_multiple(lesion_mask_cropped, multiple=16)
                print(f"Image cropped shape after padding: {image_cropped.shape}")
                print(f"Lesion mask cropped shape after padding: {lesion_mask_cropped.shape}")

                # Save the cropped images and masks per slice in wandb
                
                for idx in range(num_cropped_slices):
                    slice_num = min_z + idx  # Original slice number

                    cropped_image_slice = image_cropped[0, 0, idx, :, :].cpu().numpy()
                    cropped_label_slice = lesion_mask_cropped[0, 0, idx, :, :].cpu().numpy()
                    print(f"Slice {slice_num} - Image shape: {cropped_image_slice.shape}, Label shape: {cropped_label_slice.shape}")

                    # Visualización opcional
                    if count_visualize_overlap < 50:
                        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                        ax[0].imshow(cropped_image_slice, cmap='gray')
                        ax[0].axis('off')
                        ax[0].set_title(f'Slice {slice_num}')

                        ax[1].imshow(prostate_mask_cropped[idx, :, :], cmap='gray')
                        ax[1].axis('off')
                        ax[1].set_title(f'Prostate Mask {slice_num}')

                        ax[2].imshow(cropped_label_slice, cmap='gray')
                        ax[2].axis('off')
                        ax[2].set_title(f'Lesion Mask {slice_num}')

                        plt.tight_layout()
                        # Guardar la visualización en wandb
                        wandb.log({"Validation Cropped Images and Masks": [wandb.Image(fig)]})
                        #plt.savefig(f'/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/cropped_imgs/visualization/visualization_patient{i}_slice{slice_num}.png')
                        plt.close()

                        count_visualize_overlap += 1

            else:
                print(f"No prostate found for patient {i}")
                continue  # Pasamos al siguiente paciente
            
            optimizer.zero_grad()
            output = model(image_cropped)
            print(f"Output shape: {output.shape}")
            print(f"Range of values in output: {output.min()} - {output.max()}")
            # make the labels occupy less memory
            del image_cropped

            # Visualization of the output of the model after softmax, the predicted label map for each class (in each channel)
            if epoch % 20 == 0 and i == 0: # visualize only the first image of the first batch
                if output.max() > 1 and output.min() <= 0:
                    #softmax_output = F.softmax(output, dim=1)
                    sigmoid_output = F.sigmoid(output)
                    print(f"Applied the softmax for the output of image {i} in epoch {epoch}")
                    print(f"sigmoid_output shape: {sigmoid_output.shape}")
                else:
                    print(f"Output of image {i} in epoch {epoch} is already in the form of probabilities")
                    #softmax_output = output
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
                
                # Guardar la trama en wandb
                wandb.log({"ValidationSet Predicted Label Map": [wandb.Image(plt, caption="Predicted Label Map")]})
                print("Visualization of the predicted label maps in the validation set for each channel in wandb")
                
                # Cerrar la trama
                plt.close()
            #####################################################################################################################################
            # apply sigmoid to the output of the model
            output = F.sigmoid(output)
            print(f"Output shape: {output.shape}")
            print(f"Range of values in output after sigmoid: {output.min()} - {output.max()}")
            
            print("get same shape for the ground truth as the output of the model")
            #print(f"Ground truth shape: {ground_truth.shape}")
            lesion_mask_cropped = lesion_mask_cropped.squeeze(1) # torch.Size([1, 32, 320, 320])
            one_hot_gt = F.one_hot(lesion_mask_cropped, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            print(f"Ground truth shape: {one_hot_gt.shape}")   
              
            loss = criterion(output, one_hot_gt)
            print('Validation Loss: ', loss)
            # wandb.log({"Eval Loss": loss.item()})

            # Calculate Dice for foreground and background
            background_mask = (lesion_mask_cropped == 0).float()
            foreground_mask = (lesion_mask_cropped == 1).float()

            dice_background = dice_coef(output, background_mask)
            dice_foreground = dice_coef(output, foreground_mask)

            print(f"Dice Foreground: {dice_foreground}")
            print(f"Dice Background: {dice_background}")

            val_loss += loss.item()
            val_dice_foreground += dice_foreground.item()
            val_dice_background += dice_background.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    wandb.log({"Avg Val Loss": avg_val_loss})
    print(f"Average validation loss: {avg_val_loss}")

    avg_val_dice_foreground = val_dice_foreground / len(val_dataloader)
    avg_val_dice_background = val_dice_background / len(val_dataloader)
    wandb.log({"Avg Val Dice Foreground": avg_val_dice_foreground})
    print(f"Average validation dice foreground: {avg_val_dice_foreground}")
    wandb.log({"Avg Val Dice Background": avg_val_dice_background})
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
        #wandb.save(best_model_path)
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
