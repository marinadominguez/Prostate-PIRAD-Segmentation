import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from HOSP_pipeline.preprocessing.dataset_hosp import get_train_val_test_Dataloaders  # Ensure this is the correct import
from HOSP_pipeline.models.unet3d import UNet3D
from HOSP_pipeline.config.config import IN_CHANNELS, CHECKPOINT_PATH, WANDB_TEST_NAME, SPLIT_SEED, ALPHA #, DICE_WEIGHT, FOREGROUND_WEIGHT
from HOSP_pipeline.utils.utils import resize_image_and_label, plot_patient_slices, plot_confusion_matrix
from HOSP_pipeline.training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np
import random
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from matplotlib.colors import LogNorm
import torch.nn.functional as F
from HOSP_pipeline.utils.plot import visualize_slices_labels_prediction_overlap, visualize_slice_predictions
from HOSP_pipeline.config.config import config
#import segmentation_models_pytorch_3d as smp
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label, find_contours
import matplotlib.pyplot as plt
from skimage import segmentation
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

wandb.init(project="hops_pipeline", entity="marina-dominguez", name="Segmentation_Analysis_withCCM")

# Log the entire config dictionary
wandb.config.update(config)

# Now you can access it like this:
print("Dataset path:", wandb.config.dataset_path)
print("Number of epochs:", wandb.config.training_epoch)

# Ensure NUM_CLASSES matches the pretrained model's configuration
PRETRAINED_NUM_CLASSES = 2  # Set to 2 equal to the number of classes in the trained model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def get_largest_non_edge_component(binary_mask):
    """
    Function to get the largest connected component not touching the edges using skimage.
    """
    # Remove connected components that touch the border
    cleared_mask = clear_border(binary_mask)
    print(f"cleared_mask shape: {cleared_mask.shape}") # cleared_mask shape: (320, 320)

    # Label remaining components
    labeled_mask = label(cleared_mask)
    print(f"labeled_mask shape: {labeled_mask.shape}")
    
    # Measure properties of labeled regions
    regions = regionprops(labeled_mask)
    print(f"Number of regions: {len(regions)}")
    
    if not regions:
        return binary_mask  # No regions left after clearing borders

    # Find the largest region by area
    largest_region = max(regions, key=lambda r: r.area)
    print(f"Largest region area: {largest_region.area}")
    
    # Create a new binary mask with only the largest component
    largest_component_mask = (labeled_mask == largest_region.label).astype(int)  # Use `int` instead of `np.int`
    print(f"largest_component_mask shape: {largest_component_mask.shape}")
    
    return largest_component_mask

# Set the seed for reproducibility
def set_seed(seed=SPLIT_SEED):
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch for multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for certain functions
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior at the cost of speed

set_seed(SPLIT_SEED)  # Call this function at the start of your script

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

# MODEL
model = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)

# Load the state_dict from the checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Load the updated state_dict into the model
model.load_state_dict(checkpoint) #model_state_dict)

model = model.to(device)
model.eval()

# Initialize DataLoader
if torch.cuda.is_available():
    train_transforms = train_transform_cuda
    val_transforms = val_transform_cuda
else:
    train_transforms = train_transform
    val_transforms = val_transform
    print('CUDA not available! Training initialized on CPU ...')

_, _, test_dataloader = get_train_val_test_Dataloaders(train_transforms=train_transforms,
                                                       val_transforms=val_transforms,
                                                       test_transforms=val_transforms)

print("Test dataloader length:", len(test_dataloader))

# Initialization
all_ground_truths = []
all_predictions = []
all_images_gt = []
count_visualize_slice_predictions = 0
count_visualize_overlap = 0
iou_scores = []
threshold_selected = 0.202

# Inference
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print(f"Patient {i+1} out of {len(test_dataloader)} patients.")
        # The test dataloader returns a list with strings, so we need to convert it to a dictionary
        data = {'image': data[0], 'label': data[1]}
        image = data['image'].to(device)
        ground_truth = data['label'].to(device)
        
        resized_images, resized_labels = resize_image_and_label(image, ground_truth)
        
        resized_images = resized_images.float()
        resized_labels = resized_labels.long()

        assert resized_images.shape[2:] == resized_labels.shape[2:], \
            "Mismatch in spatial dimensions between images and labels after resizing."

        outputs = model(resized_images)
        
        # outputs_softmax = F.softmax(outputs, dim=1)
        outputs_sigmoid = torch.sigmoid(outputs)
        prostate_probabilities = outputs_sigmoid.clone()
        prostate_probabilities = prostate_probabilities[0, 1, :, :, :]
        
        predictions = (prostate_probabilities >= threshold_selected).int()

        for slice_idx in range(predictions.shape[0]): # predictions.shape[0] = 32
            predictions[slice_idx, :, :] = torch.tensor(
                #get_largest_non_edge_component(predictions[slice_idx, :, :].cpu().numpy())
                get_largest_non_near_edge_component(predictions[slice_idx, :, :].cpu().numpy()) 
            ).to(device)
        
        for slice in range(resized_labels.shape[2]):
            gt_image = resized_images[0, 0, slice, :, :].cpu().numpy()
            gt_slice = resized_labels[0, 0, slice, :, :].cpu().numpy()
            pred_slice = predictions[slice, :, :].cpu().numpy()

            all_ground_truths.append(gt_slice)
            all_predictions.append(pred_slice)
            all_images_gt.append(gt_image)

            # Only evaluate slices with prostate present
            if np.any(gt_slice > 0):
                bbox = extract_prostate_region(gt_slice)
                
                if bbox is not None:
                    min_y, max_y, min_x, max_x = bbox
                    
                    # Crop ground truth and prediction using the bounding box
                    gt_cropped = gt_slice[min_y-10:max_y+10, min_x-10:max_x+10]
                    pred_cropped = pred_slice[min_y-10:max_y+10, min_x-10:max_x+10]
                    image_cropped = gt_image[min_y-10:max_y+10, min_x-10:max_x+10]
                    
                    # Calculate IoU for this slice
                    iou = jaccard_score(gt_cropped.flatten(), pred_cropped.flatten())
                    print(f"Slice {slice} IoU: {iou}")
                    iou_scores.append(iou)

                if count_visualize_overlap < 30:
                    print(f"image cropped shape: {image_cropped.shape}")
                    print(f"gt cropped shape: {gt_cropped.shape}")
                    print(f"pred cropped shape: {pred_cropped.shape}")
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(image_cropped, cmap='gray')
                    ax[0].axis('off')
                    ax[0].set_title('Slice ' + str(slice))

                    ax[1].imshow(gt_cropped, cmap='gray')
                    ax[1].axis('off')
                    ax[1].set_title('Label ' + str(slice))

                    ax[2].imshow(pred_cropped, cmap='gray')
                    ax[2].axis('off')
                    ax[2].set_title('Prediction ' + str(slice))

                    plt.tight_layout()
                    
                    # Save and log the plot
                    generated_plot = plt.gcf()
                    wandb.log({f"Slice {slice}": wandb.Image(generated_plot)})
                    plt.close()

                    count_visualize_overlap += 1

# Create a table to store all metrics
metrics_table = wandb.Table(columns=["Metric", "Value"])

# Calculate average IoU across all slices
average_iou = np.mean(iou_scores)
metrics_table.add_data("Average IoU for Prostate Region", average_iou)

# False Positive and False Negative Rates
false_positive_rates = []
false_negative_rates = []

for i, (gt_slice, pred_slice) in enumerate(zip(all_ground_truths, all_predictions)):
    # Calculate FP and FN rates
    false_positives = np.sum((pred_slice == 1) & (gt_slice == 0))
    false_negatives = np.sum((pred_slice == 0) & (gt_slice == 1))
    total_gt = np.sum(gt_slice == 1)  # Total prostate pixels

    # Avoid division by zero
    fn_rate = false_negatives / total_gt if total_gt > 0 else 0
    total_pred = np.sum(pred_slice == 1)  # Total predicted prostate pixels
    fp_rate = false_positives / total_pred if total_pred > 0 else 0

    false_positive_rates.append(fp_rate)
    false_negative_rates.append(fn_rate)

# Plot ROC Curve for FP vs FN rates
plt.figure(figsize=(10, 5))
plt.plot(false_positive_rates, false_negative_rates, marker="o", color="skyblue")
plt.xlabel("False Positive Rate")
plt.ylabel("False Negative Rate")
plt.title("ROC Curve")
wandb.log({"ROC Curve": wandb.Image(plt)})
plt.close()

# IoU Distribution
plt.figure(figsize=(10, 5))
plt.hist(iou_scores, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("IoU Score")
plt.ylabel("Frequency")
plt.title("Distribution of IoU Scores per Slice")
wandb.log({"IoU Distribution": wandb.Image(plt)})
plt.close()

# IoU by Slice Index
slice_indices = list(range(len(iou_scores)))  # Assuming one IoU per slice
plt.figure(figsize=(10, 5))
plt.plot(slice_indices, iou_scores, marker="o", color="skyblue")
plt.xlabel("Slice Index")
plt.ylabel("IoU Score")
plt.title("IoU Score by Slice Index")
wandb.log({"IoU by Slice Index": wandb.Image(plt)})
plt.close()

# Calculate Dice Score for each slice with zero_division handling
dice_scores = [f1_score(gt.flatten(), pred.flatten(), zero_division=0) for gt, pred in zip(all_ground_truths, all_predictions)]
average_dice = np.mean(dice_scores)
metrics_table.add_data("Average Dice Score for Prostate Region", average_dice)


# Volume Difference
gt_volume = np.sum(all_ground_truths)  # Total volume in the ground truth
pred_volume = np.sum(all_predictions)  # Total volume in predictions
volume_difference = abs(gt_volume - pred_volume) / gt_volume * 100  # Percentage difference
metrics_table.add_data("Volume Difference (%)", volume_difference)

def calculate_hausdorff_distance(gt, pred):
    # Extract boundaries
    gt_contours = find_contours(gt, 0.5)
    pred_contours = find_contours(pred, 0.5)
    
    # Check if contours were found in both ground truth and prediction
    if len(gt_contours) == 0 or len(pred_contours) == 0:
        # Return a default value if either contour list is empty
        # You could return np.inf or some large number to indicate no overlap
        return np.inf
    
    # Flatten the contours to a list of points
    gt_points = np.concatenate(gt_contours, axis=0)
    pred_points = np.concatenate(pred_contours, axis=0)
    
    # Calculate directed Hausdorff distance in both directions
    hd_gt_to_pred = directed_hausdorff(gt_points, pred_points)[0]
    hd_pred_to_gt = directed_hausdorff(pred_points, gt_points)[0]
    
    # Symmetric Hausdorff distance
    hd = max(hd_gt_to_pred, hd_pred_to_gt)
    return hd

hausdorff_distances = []
for gt_slice, pred_slice in zip(all_ground_truths, all_predictions):
    if np.any(gt_slice):  # Only compute if there's something in the ground truth
        hd = calculate_hausdorff_distance(gt_slice, pred_slice)
        hausdorff_distances.append(hd)

average_hd = np.mean(hausdorff_distances)
metrics_table.add_data("Average Hausdorff Distance", average_hd)

# Add the metrics table to WandB
wandb.log({"Metrics Table": metrics_table})

"""for i in range(len(iou_scores)):  # Loop over the actual length of iou_scores
    plt.figure(figsize=(5, 5))
    plt.title(f"Slice {i} - IoU: {iou_scores[i]:.3f}")
    plt.axis('off')
    wandb.log({f"Overlayed Slice {i}": wandb.Image(plt)})
    plt.close()"""

# Finish the WandB run
wandb.finish()