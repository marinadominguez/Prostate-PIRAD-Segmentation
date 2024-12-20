import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..preprocessing.dataset_hosp import get_train_val_test_Dataloaders
from ..models.unet3d import UNet3D
from ..config.config import IN_CHANNELS, TEST_PATIENT_IDS, PROSTATE_CHECKPOINT
from ..utils.utils import resize_image_and_label
from ..training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np

from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from PIL import Image
import json
import pydicom

PRETRAINED_NUM_CLASSES = 2  # Asegúrate de que coincide con el modelo entrenado
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ruta al archivo JSON con las anotaciones de las lesiones
annotations_path = 'preprocessing/interpolation_preprocess/interpolated_ALL_annotated_cancer_december_minannot1_maxgap6.json'

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

# Cargar el modelo
pretrained_prostate_model = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)
checkpoint = torch.load(PROSTATE_CHECKPOINT, map_location=device)
pretrained_prostate_model.load_state_dict(checkpoint)
pretrained_prostate_model = pretrained_prostate_model.to(device)
pretrained_prostate_model.eval()

# Inicializar DataLoader
if torch.cuda.is_available():
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
else:
    train_transforms = train_transform
    val_transforms = val_transform
    print('CUDA not available! Training initialized on CPU ...')

_, _, test_dataloader = get_train_val_test_Dataloaders(train_transforms=train_transforms,
                                                       val_transforms=val_transforms,
                                                       test_transforms=val_transforms,
                                                       test_ids=TEST_PATIENT_IDS,)
print("Test dataloader length:", len(test_dataloader)) 
# i have set the split to 0, 0, 1, so i only have test, because i want to apply this in the whole new lessions dataset

# Directorios para guardar las imágenes y etiquetas recortadas
cropped_image_dir = 'dataset/cropped_imgs/images' 
cropped_label_dir = 'dataset/cropped_imgs/labels'  
os.makedirs(cropped_image_dir, exist_ok=True)
os.makedirs(cropped_label_dir, exist_ok=True)

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

# Inference
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print(f"Patient {i+1} out of {len(test_dataloader)} patients.")

        # Desempaquetar los datos
        image = data[0].to(device)
        ground_truth = data[1].to(device)

        # Generar un folder_id único basado en el índice del paciente
        folder_id = f'patient_{i}'

        resized_images, resized_labels = resize_image_and_label(image, ground_truth)
        resized_images = resized_images.float()
        resized_labels = resized_labels.long()

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
            image_cropped = resized_images[0, 0, min_z:max_z+1, min_y:max_y+1, min_x:max_x+1].cpu().numpy()
            lesion_mask_cropped = resized_labels[0, 0, min_z:max_z+1, min_y:max_y+1, min_x:max_x+1].cpu().numpy()
            prostate_mask_cropped = prostate_mask[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

            # Save the cropped images and masks per slice
            num_cropped_slices = image_cropped.shape[0]
            for idx in range(num_cropped_slices):
                slice_num = min_z + idx  # Original slice number

                cropped_image_slice = image_cropped[idx, :, :]
                cropped_label_slice = lesion_mask_cropped[idx, :, :]

                cropped_image_path = os.path.join(cropped_image_dir, f'{folder_id}_slice_{slice_num}.png')
                cropped_label_path = os.path.join(cropped_label_dir, f'{folder_id}_slice_{slice_num}_label.png')

                # Process and save the cropped image
                if cropped_image_slice.dtype != np.uint8:
                    # Scale if necessary; adjust scaling based on your data
                    image_uint8 = (cropped_image_slice * 255).astype(np.uint8)
                else:
                    image_uint8 = cropped_image_slice

                cropped_image = Image.fromarray(image_uint8, mode='L')  # Use 'RGB' if it's a color image
                cropped_image.save(cropped_image_path)

                # Process and save the cropped label
                if cropped_label_slice.dtype != np.uint8:
                    label_uint8 = cropped_label_slice.astype(np.uint8)
                else:
                    label_uint8 = cropped_label_slice

                cropped_label = Image.fromarray(label_uint8, mode='L')
                cropped_label.save(cropped_label_path)

                # Actualizar las nuevas anotaciones
                new_annotation = {
                    'slice': cropped_image_path,
                    'label': cropped_label_path,
                    'Folder_id': folder_id
                    #'slice_num': int(slice_num)
                }
                new_annotations.append(new_annotation)

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
                    # Guardar la visualización
                    plt.savefig(f'dataset/cropped_imgs/visualization/visualization_{folder_id}_slice{slice_num}.png')
                    plt.close()

                    count_visualize_overlap += 1

        else:
            print(f"No prostate found for patient {folder_id}")
            continue  # Pasamos al siguiente paciente
    
    # Guardar las nuevas anotaciones
    new_annotations_path = 'preprocessing/cropping_imgs_prostates&lessions/nuevas_anotaciones_recortadas.json'
    with open(new_annotations_path, 'w') as f:
        json.dump(new_annotations, f)

    print(f"Se guardaron {len(new_annotations)} anotaciones recortadas en {new_annotations_path}")
