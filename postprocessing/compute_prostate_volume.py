import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
from HOSP_pipeline.preprocessing.dataset_hosp import get_train_val_test_Dataloaders  # Ensure this is the correct import
from HOSP_pipeline.models.unet3d import UNet3D
from HOSP_pipeline.config.config import IN_CHANNELS, CHECKPOINT_PATH, WANDB_TEST_NAME, SPLIT_SEED
from HOSP_pipeline.utils.utils import resize_image_and_label, plot_patient_slices, plot_confusion_matrix
from HOSP_pipeline.training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from matplotlib.colors import LogNorm
import torch.nn.functional as F
from HOSP_pipeline.utils.plot import visualize_slices_labels_prediction_overlap, visualize_slice_predictions
from HOSP_pipeline.config.config import config
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from HOSP_pipeline.utils.utils import get_largest_non_edge_component, get_largest_non_near_edge_component

wandb.init(project="hops_pipeline", entity="marina-dominguez", name="1x1mm_VolumeComputation_onlyLabelledProstates")

# Log the entire config dictionary
wandb.config.update(config)

# Now you can access it like this:
print("Dataset path:", wandb.config.dataset_path)
print("Number of epochs:", wandb.config.training_epoch)
threshold_selected = 0.202 #0.6062 #0.0185 #0.716 #0.147 # #0.0368 

# Ensure NUM_CLASSES matches the pretrained model's configuration
PRETRAINED_NUM_CLASSES = 2  # Set to 2 equal to the number of classes in the trained model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# MODEL
model = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)

# Load the state_dict from the checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Get the current model's state_dict
#model_state_dict = model.state_dict()

# Filter out mismatched keys
#filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

# Update the model's state_dict with the matching parameters
#model_state_dict.update(filtered_checkpoint)

# Load the updated state_dict into the model
model.load_state_dict(checkpoint)#model_state_dict)

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

# Metadata for voxel dimensions
pixel_spacing_x = 1.0 #0.5  # mm
pixel_spacing_y = 1.0 #0.5  # mm
slice_thickness = 3.0  # mm
slice_location_difference = 3.2999  # mm (computed from DICOM Slice Location differences)

# Voxel volume formulas
measured_voxel_volume = pixel_spacing_x * pixel_spacing_y * slice_thickness
contiguous_voxel_volume = pixel_spacing_x * pixel_spacing_y * slice_location_difference

print(f"Measured Voxel Volume: {measured_voxel_volume:.6f} mm³")
print(f"Contiguous Voxel Volume: {contiguous_voxel_volume:.6f} mm³")

# Function to compute volume
def compute_volume(predictions, voxel_volume):
    """
    Compute the prostate volume based on predictions and voxel volume.

    Args:
    - predictions (np.ndarray): 3D binary mask (slices, rows, cols) of segmentation.
    - voxel_volume (float): Volume of one voxel in mm³.

    Returns:
    - float: Total volume in mm³.
    """
    total_voxels = np.sum(predictions)  # Count non-zero (foreground) voxels
    total_volume = total_voxels * voxel_volume
    return total_volume

# Initialization
all_ground_truths = []
all_predictions = []
count_predictions = 0
count_overlapping_predictions = 0
previous_predicted_prostate_volume_contiguous = 0
previous_gt_prostate_volume_measured = 0
previous_gt_prostate_volume_contiguous = 0
previous_predicted_prostate_volume_measured = 0

patient_dict_keys = ['0f9cb0851d69ad4', '3e289b5789de903', '5588d57cd979ab0', '69e9f1dd231bc70', 
                     '6f150cd21f5f7b3', '8756a03c0a9de55', '8ead5e41fd3c1e8', '92b7b541d4e68c3', 
                     'a0bf4518e29079a', 'a16d130e19a8add', 'a34a485f7966f94', 'a818f4d4f02814a', 
                     'afec244a01137c0', 'b9ac2b350ebfc15', 'badc8ec3fcf319f', 'bcf1e35d49070cf', 
                     'cd2eaece70fbd0f', 'd501f04c5337ebd', 'ebe1a437b560432']

# Initialize the WandB table
columns = [
    "Patient", 
    "Slice", 
    "Predicted Volume (Measured)", 
    "Predicted Volume (Contiguous)", 
    "Actual Volume (Measured)", 
    "Actual Volume (Contiguous)"
]
volume_table = wandb.Table(columns=columns)

# Inference
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print(f"Patient {i+1} out of {len(test_dataloader)} patients.")
        # The test dataloader returns a list with strings, so we need to convert it to a dictionary
        data = {'image': data[0], 'label': data[1]}
        print(f"Data keys: {data.keys()}")
        print(f"Data image shape: {data['image'].shape}")
        print(f"Data label shape: {data['label'].shape}")
        print(f"Patient ID: {patient_dict_keys[i]}")

        image = data['image'].to(device)
        ground_truth = data['label'].to(device)
        id_patient = patient_dict_keys[i]
        
        resized_images, resized_labels = resize_image_and_label(image, ground_truth)
        
        resized_images = resized_images.float()
        resized_labels = resized_labels.long()

        assert resized_images.shape[2:] == resized_labels.shape[2:], \
            "Mismatch in spatial dimensions between images and labels after resizing."

        outputs = model(resized_images)
        
        outputs_sigmoid = torch.sigmoid(outputs)
        outputs_argmax = outputs_sigmoid.argmax(dim=1)
        
        prostate_probabilities = outputs_sigmoid.clone()
        prostate_probabilities = prostate_probabilities[0, 1, :, :, :]
        
        #change the threshold depending on the model 
        predictions = (prostate_probabilities >= threshold_selected).int()
        # 0.0185
        # for the model trained from scratch, the threshold is 0.551
        # for the model fine-tuned, the threshold is 0.053
        # for the public model, the threshold is 0.863
        # probatina con el threshold que dice ser optimo en el postprocessing 0.000000679 (mal)
        # pre processing interpolation from scratch, threshold = 0.4409
        # pre processing with interpolation fine-tuned, threshold = 0.01847

        print(f"prostate predictions shape: {predictions.shape}")

        # Apply the preprocessing to each slice
        for slice_idx in range(predictions.shape[0]): # predictions.shape[0] = 32
            predictions[slice_idx, :, :] = torch.tensor(
                #get_largest_non_edge_component(predictions[slice_idx, :, :].cpu().numpy())
                get_largest_non_near_edge_component(predictions[slice_idx, :, :].cpu().numpy()) 
            ).to(device)

        # Iterate over slices and log data
        for slice in range(resized_labels.shape[2]):
            if torch.any(resized_labels[0, 0, slice, :, :] > 0):
                all_ground_truths.append(resized_labels[0, 0, slice, :, :].cpu().numpy())
                all_predictions.append(predictions[slice, :, :].cpu().numpy())

                slice_predicted_prostate_volume_measured = compute_volume(predictions[slice, :, :].cpu().numpy(), measured_voxel_volume)
                slice_predicted_prostate_volume_contiguous = compute_volume(predictions[slice, :, :].cpu().numpy(), contiguous_voxel_volume)
                slice_gt_prostate_volume_measured = compute_volume(resized_labels[0, 0, slice, :, :].cpu().numpy(), measured_voxel_volume)
                slice_gt_prostate_volume_contiguous = compute_volume(resized_labels[0, 0, slice, :, :].cpu().numpy(), contiguous_voxel_volume)

                print(f"Slice {slice} - Predicted Prostate Volume (Measured): {slice_predicted_prostate_volume_measured:.2f} mm³")
                print(f"Slice {slice} - Predicted Prostate Volume (Contiguous): {slice_predicted_prostate_volume_contiguous:.2f} mm³")
                print(f"Slice {slice} - Actual Prostate Volume (Measured): {slice_gt_prostate_volume_measured:.2f} mm³")
                print(f"Slice {slice} - Actual Prostate Volume (Contiguous): {slice_gt_prostate_volume_contiguous:.2f} mm³")

                # Add a row to the WandB table
                volume_table.add_data(
                    id_patient,  # Patient number
                    slice,  # Slice number
                    slice_predicted_prostate_volume_measured,
                    slice_predicted_prostate_volume_contiguous,
                    slice_gt_prostate_volume_measured,
                    slice_gt_prostate_volume_contiguous
                )

                if count_predictions < 10:
                    visualize_slice_predictions(resized_images, resized_labels, outputs, outputs_sigmoid, outputs_argmax, predictions, slice)
                    count_predictions += 1

        if count_overlapping_predictions < 10:
            visualize_slices_labels_prediction_overlap(resized_images, resized_labels, predictions, 32, 4, figsize=(10, 60))
            count_overlapping_predictions += 1

        # Compute overall patient-level volumes
        predicted_prostate_volume_measured = compute_volume(all_predictions, measured_voxel_volume)
        predicted_prostate_volume_contiguous = compute_volume(all_predictions, contiguous_voxel_volume)
        gt_prostate_volume_measured = compute_volume(all_ground_truths, measured_voxel_volume)
        gt_prostate_volume_contiguous = compute_volume(all_ground_truths, contiguous_voxel_volume)

        real_predicted_prostate_volume_measured = predicted_prostate_volume_measured - previous_predicted_prostate_volume_measured
        real_predicted_prostate_volume_contiguous = predicted_prostate_volume_contiguous - previous_predicted_prostate_volume_contiguous
        real_gt_prostate_volume_measured = gt_prostate_volume_measured - previous_gt_prostate_volume_measured
        real_gt_prostate_volume_contiguous = gt_prostate_volume_contiguous - previous_gt_prostate_volume_contiguous

        # Add patient-level summary to the table
        volume_table.add_data(
            id_patient,  # Patient number
            -1,     # Use -1 to indicate overall summary
            real_predicted_prostate_volume_measured,
            real_predicted_prostate_volume_contiguous,
            real_gt_prostate_volume_measured,
            real_gt_prostate_volume_contiguous
        )

        # The real patient-level volumes are the new computed volumes minus the previous ones, since we are adding the new patient, they are cumulative
        
        print(f"Patient {i + 1} - Predicted Prostate Volume (Measured): {real_predicted_prostate_volume_measured:.2f} mm³")
        print(f"Patient {i + 1} - Predicted Prostate Volume (Contiguous): {real_predicted_prostate_volume_contiguous:.2f} mm³")
        print(f"Patient {i + 1} - Actual Prostate Volume (Measured): {real_gt_prostate_volume_measured:.2f} mm³")
        print(f"Patient {i + 1} - Actual Prostate Volume (Contiguous): {real_gt_prostate_volume_contiguous:.2f} mm³")

        # save the previous patient-level volumes
        previous_predicted_prostate_volume_measured = predicted_prostate_volume_measured
        previous_predicted_prostate_volume_contiguous = predicted_prostate_volume_contiguous
        previous_gt_prostate_volume_measured = gt_prostate_volume_measured
        previous_gt_prostate_volume_contiguous = gt_prostate_volume_contiguous


# Log the entire table to WandB at the end
wandb.log({"Prostate Volume Table": volume_table})


# Convertir a numpy arrays
all_ground_truths = np.array(all_ground_truths) # all the slices of all the patients labeled and not labeled (label column in the image)
all_predictions = np.array(all_predictions) # all the predictions of all the slices of all the patients (prediction column in the image)

# Compute the predicted prostate volume using both voxel volume formulas
predicted_prostate_volume_measured = compute_volume(all_predictions, measured_voxel_volume)
predicted_prostate_volume_contiguous = compute_volume(all_predictions, contiguous_voxel_volume)
gt_prostate_volume_measured = compute_volume(all_ground_truths, measured_voxel_volume)
gt_prostate_volume_contiguous = compute_volume(all_ground_truths, contiguous_voxel_volume)

print(f"Predicted Prostate Volume (Measured): {predicted_prostate_volume_measured:.2f} mm³")
print(f"Predicted Prostate Volume (Contiguous): {predicted_prostate_volume_contiguous:.2f} mm³")
print(f"Actual Prostate Volume (Measured): {gt_prostate_volume_measured:.2f} mm³")
print(f"Actual Prostate Volume (Contiguous): {gt_prostate_volume_contiguous:.2f} mm³")

wandb.finish()