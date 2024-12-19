import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..preprocessing.dataset_hosp import get_train_val_test_Dataloaders  # Ensure this is the correct import
from ..models.unet3d import UNet3D
from ..config.config import IN_CHANNELS, CHECKPOINT_PATH, WANDB_TEST_NAME
from ..utils.utils import resize_image_and_label, plot_patient_slices, plot_confusion_matrix
from ..training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from matplotlib.colors import LogNorm
import torch.nn.functional as F
from ..utils.plot import visualize_slices_labels_prediction_overlap, visualize_slice_predictions
from ..config.config import config

wandb.init(project="hops_pipeline", entity="marina-dominguez", name="trained_from_scratch_testing-Threshold-FPR0.15-plot-0.4")
# Log the entire config dictionary
wandb.config.update(config)
# Now you can access it like this:
print("Dataset path:", wandb.config.dataset_path)
print("Number of epochs:", wandb.config.training_epoch)

# Ensure NUM_CLASSES matches the pretrained model's configuration
PRETRAINED_NUM_CLASSES = 2  # Set to 2 equal to the number of classes in the trained model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL
"""
model = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()
"""
# MODEL
model = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)

# Load the state_dict from the checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Get the current model's state_dict
model_state_dict = model.state_dict()

# Filter out mismatched keys
filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

# Update the model's state_dict with the matching parameters
model_state_dict.update(filtered_checkpoint)

# Load the updated state_dict into the model
model.load_state_dict(model_state_dict)

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
all_argmax_predictions = []
all_softmax_predictions_class1 = []

# Inference
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print(f"Patient {i+1} out of {len(test_dataloader)} patients.")
        # The test dataloader returns a list with strings, so we need to convert it to a dictionary
        data = {'image': data[0], 'label': data[1]}
        print(f"Data keys: {data.keys()}")
        print(f"Data image shape: {data['image'].shape}")
        print(f"Data label shape: {data['label'].shape}")

        image = data['image'].to(device)
        ground_truth = data['label'].to(device)
        
        resized_images, resized_labels = resize_image_and_label(image, ground_truth)
        
        resized_images = resized_images.float()
        resized_labels = resized_labels.long()

        assert resized_images.shape[2:] == resized_labels.shape[2:], \
            "Mismatch in spatial dimensions between images and labels after resizing."

        outputs = model(resized_images)
        
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1)
        
        prostate_probabilities = outputs_softmax.clone()
        prostate_probabilities = prostate_probabilities[0, 1, :, :, :]
        
        predictions = (prostate_probabilities >= 0.4).int()
        
        # Filtrado de slices sin etiquetas
        for slice in range(resized_labels.shape[2]):
            # Solo consideramos la slice si tiene alguna etiqueta diferente de fondo
            if torch.any(resized_labels[0, 0, slice, :, :] > 0):
                # Guardamos las imágenes, etiquetas y predicciones solo para slices etiquetadas
                all_ground_truths.append(resized_labels[0, 0, slice, :, :].cpu().numpy())
                all_argmax_predictions.append(outputs_argmax[0, slice, :, :].cpu().numpy())
                all_softmax_predictions_class1.append(outputs_softmax[0, 1, slice, :, :].cpu().numpy())

                # Visualización opcional, solo para slices etiquetadas
                visualize_slice_predictions(resized_images, resized_labels, outputs, outputs_softmax, outputs_argmax, predictions, slice)
        
        # Visualización opcional, para todas las slices
        # arguments : resized_images, resized_labels, outputs, num_rows, num_cols, figsize=(10, 60)
        visualize_slices_labels_prediction_overlap(resized_images, resized_labels, outputs_argmax, 32, 4)

# Convertir a numpy arrays
all_ground_truths = np.array(all_ground_truths)
all_argmax_predictions = np.array(all_argmax_predictions)
all_softmax_predictions_class1 = np.array(all_softmax_predictions_class1)

# Aplanar para cálculo de métricas
gt_flat = all_ground_truths.flatten()
pred_flat = all_argmax_predictions.flatten()

# Filtramos las etiquetas de fondo
foreground_indices = gt_flat > 0

gt_foreground = gt_flat[foreground_indices]
pred_foreground = pred_flat[foreground_indices]

precision = precision_score(gt_flat, pred_flat, pos_label=1, average='weighted')
print("Precision:", precision)

iou = jaccard_score(gt_foreground, pred_foreground, pos_label=1, average='weighted')
print("foreground IoU (Jaccard Index):", iou)
iou = jaccard_score(gt_flat, pred_flat, pos_label=1, average='weighted')
print("IoU (Jaccard Index):", iou)

accuracy = accuracy_score(gt_foreground, pred_foreground)
print("foreground Accuracy:", accuracy)
accuracy = accuracy_score(gt_flat, pred_flat)
print("Accuracy:", accuracy)

f1 = f1_score(gt_foreground, pred_foreground, pos_label=1, average='weighted')
print("F1 Score:", f1)
#f1 = f1_score(gt_flat, pred_flat, pos_label=1, average='weighted')
#print("F1 Score:", f1)

tabla_metricas = wandb.Table(columns=["Precision", "IoU (Jaccard Index)", "Accuracy", "F1 Score"])
tabla_metricas.add_data(precision, iou, accuracy, f1)
wandb.log({"Metrics": tabla_metricas})

true_binary_labels = all_ground_truths.flatten()
target_scores = all_softmax_predictions_class1.flatten()

tpr, fpr, th = roc_curve(true_binary_labels, target_scores)
auc_score = roc_auc_score(true_binary_labels, target_scores)
print(f"AUC: {auc_score}")

plt.figure(figsize=(10, 7))
plt.plot(tpr, fpr, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

wandb.log({"ROC Curve": wandb.Image(plt)})
plt.close()
print("ROC curve plotted")

fpr_threshold = 0.15
threshold_index = np.argmax(fpr >= fpr_threshold)
threshold = th[threshold_index]

print(f"Threshold for FPR = {fpr_threshold}: {threshold}")
wandb.log({"Threshold for FPR 0.15": threshold})

wandb.finish()