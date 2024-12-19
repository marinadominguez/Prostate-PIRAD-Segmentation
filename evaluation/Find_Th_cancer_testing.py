import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..preprocessing.dataset_hosp import get_train_val_test_Dataloaders  # Ensure this is the correct import
from ..models.unet3d import UNet3D
from ..config.config import IN_CHANNELS, CHECKPOINT_PATH, WANDB_TEST_NAME, SPLIT_SEED, ALPHA, PROSTATE_CHECKPOINT, TEST_PATIENT_IDS, TRAIN_PATIENTS_IDS, VAL_PATIENTS_IDS #, DICE_WEIGHT, FOREGROUND_WEIGHT
from ..utils.utils import resize_image_and_label, plot_patient_slices, plot_confusion_matrix
from ..training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np
import random
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from matplotlib.colors import LogNorm
import torch.nn.functional as F
from ..utils.plot import visualize_slices_labels_prediction_overlap, visualize_slice_predictions
from ..config.config import config
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
#import segmentation_models_pytorch_3d as smp


wandb.init(project="hops_pipeline", entity="marina-dominguez", name = f"from_scratch_find_threshold_{ALPHA}alphaCustomFocal") #_{DICE_WEIGHT}weightDice_{FOREGROUND_WEIGHT}ForegrWeight")
           # name="Find_Threshold_PubliclyTrainedNetwork_0.7_5-5")

# Log the entire config dictionary
wandb.config.update(config)

# Now you can access it like this:
print("Dataset path:", wandb.config.dataset_path)
print("Number of epochs:", wandb.config.training_epoch)

# Ensure NUM_CLASSES matches the pretrained model's configuration
PRETRAINED_NUM_CLASSES = 2  # Set to 2 equal to the number of classes in the trained model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pad_to_multiple(tensor, multiple=16):
    depth, height, width = tensor.shape[-3:]
    pad_depth = (multiple - depth % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    padding = (0, pad_width, 0, pad_height, 0, pad_depth)  # Padding in the order (W, H, D)
    padded_tensor = F.pad(tensor, padding)
    return padded_tensor

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

# Cargar el modelo pre-entrenado de la próstata
pretrained_prostate_model = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)
checkpoint = torch.load(PROSTATE_CHECKPOINT, map_location=device)
pretrained_prostate_model.load_state_dict(checkpoint)
pretrained_prostate_model = pretrained_prostate_model.to(device)
pretrained_prostate_model.eval()

# MODEL
model_cancer = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)
# Load the state_dict from the checkpoint
checkpoint_cancer = torch.load(CHECKPOINT_PATH, map_location=device)
# Get the current model's state_dict
#model_state_dict = model.state_dict()
# Filter out mismatched keys
#filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
# Update the model's state_dict with the matching parameters
#model_state_dict.update(filtered_checkpoint)
# Load the updated state_dict into the model
model_cancer.load_state_dict(checkpoint_cancer) #model_state_dict)
model_cancer = model_cancer.to(device)
model_cancer.eval()

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
                                                       test_transforms=val_transforms,
                                                       train_ids=TRAIN_PATIENTS_IDS,
                                                       val_ids=VAL_PATIENTS_IDS,
                                                       test_ids=TEST_PATIENT_IDS)

print("Test dataloader length:", len(test_dataloader))

# Initialization
all_ground_truths = []
all_argmax_predictions = []
#all_predictions = []
all_softmax_predictions_class1 = []
count_visualize_slice_predictions = 0
count_visualize_overlap = 0
threshold_selected = 0.3
# Inicializar variables para almacenar las dimensiones máximas
max_height = 0
max_width = 0


# Inference
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print(f"Patient {i+1} out of {len(test_dataloader)} patients.")
        data = {'image': data[0], 'label': data[1]}
        image = data['image'].to(device)
        ground_truth = data['label'].to(device)
        
        resized_images, resized_labels = resize_image_and_label(image, ground_truth)
        resized_images = resized_images.float()
        resized_labels = resized_labels.long()

        # Segmentación de la próstata usando el modelo pre-entrenado
        outputs = pretrained_prostate_model(resized_images)
        outputs_sigmoid = torch.sigmoid(outputs)
        prostate_probabilities = outputs_sigmoid[0, 1, :, :, :]
        predictions = (prostate_probabilities >= threshold_selected).int()

        # Post-procesamiento de la segmentación de la próstata
        for slice_idx in range(predictions.shape[0]):
            predictions[slice_idx, :, :] = torch.tensor(
                get_largest_non_near_edge_component(predictions[slice_idx, :, :].cpu().numpy())
            ).to(device)

        prostate_mask = predictions.cpu().numpy()
        bbox = extract_prostate_region_3d(prostate_mask)

        if bbox is not None:
            min_z, max_z, min_y, max_y, min_x, max_x = bbox
            # Aplicar márgenes
            margin_z = 0
            margin_xy = 10
            min_z = max(min_z - margin_z, 0)
            max_z = min(max_z + margin_z, prostate_mask.shape[0] - 1)
            min_y = max(min_y - margin_xy, 0)
            max_y = min(max_y + margin_xy, prostate_mask.shape[1] - 1)
            min_x = max(min_x - margin_xy, 0)
            max_x = min(max_x + margin_xy, prostate_mask.shape[2] - 1)

            # Recortar las imágenes y las etiquetas usando el bounding box
            image_cropped = resized_images[:, :, min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
            label_cropped = resized_labels[:, :, min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

            # Padding para ajustar el tamaño a múltiplos de 16
            image_cropped = pad_to_multiple(image_cropped, multiple=16)
            label_cropped = pad_to_multiple(label_cropped, multiple=16)
        else:
            print(f"No prostate found for patient {i}")
            continue  # Pasamos al siguiente paciente

        for slice in range(label_cropped.shape[2]):
            if torch.any(label_cropped[0, 0, slice, :, :] > 0):
                h, w = label_cropped[0, 0, slice, :, :].shape[-2:]
                if h > max_height:
                    max_height = h
                if w > max_width:
                    max_width = w
        
        # Realizar la inferencia con el modelo entrenado
        outputs = model_cancer(image_cropped)
        # outputs_softmax = F.softmax(outputs, dim=1)
        outputs_sigmoid = torch.sigmoid(outputs)
        print(f"outputs_sigmoid shape: {outputs_sigmoid.shape}") # outputs_sigmoid shape: torch.Size([1, 2, 32, 320, 320])
        outputs_argmax = outputs_sigmoid.argmax(dim=1)
        
        prostate_probabilities = outputs_sigmoid.clone()
        prostate_probabilities = prostate_probabilities[0, 1, :, :, :]
        
        predictions = (prostate_probabilities >= threshold_selected).int()
        #print("Predictions shape:", predictions.shape) #Predictions shape: torch.Size([32, 320, 320])
        
        for slice in range(label_cropped.shape[2]):
            if torch.any(label_cropped[0, 0, slice, :, :] > 0):
                # Solo consideramos la slice si tiene alguna etiqueta diferente de fondo
                h, w = label_cropped[0, 0, slice, :, :].shape[-2:]
                print(f"Heigth: {h}, Width: {w}")
                pad_height = max_height - h
                pad_width = max_width - w
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                print(f"Pad top: {pad_top}, Pad bottom: {pad_bottom}, Pad left: {pad_left}, Pad right: {pad_right}")

                # Extraer y padear ground truth
                ground_truth_slice = label_cropped[0, 0, slice, :, :].cpu().numpy()
                padded_gt_slice = np.pad(
                    ground_truth_slice,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=0
                )

                # Verificar la forma
                print(f"Padded ground truth shape: {padded_gt_slice.shape}")

                # Agregar a la lista
                all_ground_truths.append(padded_gt_slice)

                # Extraer y padear predicción
                prediction_slice = outputs_sigmoid[0, 1, slice, :, :].cpu().numpy()
                padded_pred_slice = np.pad(
                    prediction_slice,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=0
                )

                # Verificar la forma
                print(f"Padded prediction shape: {padded_pred_slice.shape}")

                # Agregar a la lista
                all_softmax_predictions_class1.append(padded_pred_slice)
                
                print(f"Slice {slice} processed.") # Slice 31 processed.
            #print("Image shape:", resized_images[0, 0, slice, :, :].shape) # Image shape: torch.Size([320, 320])
            #print("Ground truth shape:", resized_labels[0, 0, slice, :, :].shape) # Ground truth shape: torch.Size([320, 320])
            #print("Output shape:", outputs[0, :, slice, :, :].shape) # Output shape: torch.Size([2, 320, 320])
            #print("Prostate probabilities shape:", outputs_softmax[0, 1, slice, :, :].shape) # Prostate probabilities shape: torch.Size([320, 320])
            #print("Prediction shape:", outputs_argmax[0, slice, :, :].shape) # Prediction shape: torch.Size([320, 320])
            #print("Prediction shape:", predictions.shape) # Prediction shape: torch.Size([32, 320, 320])
            # Solo consideramos la slice si tiene alguna etiqueta diferente de fondo
            #if torch.any(resized_labels[0, 0, slice, :, :] > 0):
                if count_visualize_slice_predictions < 10: 
                    visualize_slice_predictions(image_cropped, label_cropped, outputs, outputs_sigmoid, outputs_argmax, predictions, slice)
                    count_visualize_slice_predictions += 1
        # Visualización para todas las slices
        # arguments : resized_images, resized_labels, outputs, num_rows, num_cols, figsize=(10, 60)
        #print("Visualizing all slices")
        #print("Image shape:", resized_images[0, 0, :, :, :].shape) # Image shape: torch.Size([32, 320, 320])
        #print("Ground truth shape:", resized_labels[0, 0, :, :, :].shape) # Ground truth shape: torch.Size([32, 320, 320])
        #print("Output shape:", outputs_argmax[0, :, :, :].shape) # Output shape: torch.Size([32, 320, 320])
        # print("predictions shape: ", predictions.shape) # predictions shape:  torch.Size([32, 320, 320])
        num_rows = image_cropped.shape[2]
        if count_visualize_overlap < 10:
            visualize_slices_labels_prediction_overlap(image_cropped, label_cropped, predictions, num_rows, 4, figsize=(10, 60))
            count_visualize_overlap += 1

# Verificar las formas
print("Verificando formas de las slices...")
for idx, gt_slice in enumerate(all_ground_truths):
    print(f"all_ground_truths[{idx}] shape: {gt_slice.shape}")
for idx, pred_slice in enumerate(all_softmax_predictions_class1):
    print(f"all_softmax_predictions_class1[{idx}] shape: {pred_slice.shape}")

# Convertir las listas en arrays de NumPy
all_ground_truths = np.array(all_ground_truths)
all_softmax_predictions_class1 = np.array(all_softmax_predictions_class1)

print(f"All ground truths shape: {all_ground_truths.shape}")
print(f"All predictions shape: {all_softmax_predictions_class1.shape}")

gt_flat = all_ground_truths.flatten()
print("Ground truth flattened shape:", gt_flat.shape) 
# Ground truth flattened shape: (3276800,)
#pred_flat = all_predictions.flatten()

# Matriz de confusión
print("CALCULO PARA TODOS LOS THRESHOLDS")

# Inicializar la tabla
tabla_metricas_manuales = wandb.Table(columns=["TPR", "TNR", "PPV", "NPV", "IoU clase 0", "IoU clase 1", "Threshold"])

# Crear el rango de thresholds
thresholds = np.linspace(0.0001, 0.9, 50)

iou_scores = []
for th in thresholds:
    # Aplicar el umbral a las probabilidades de próstata (softmax de clase 1)
    pred_flat = (all_softmax_predictions_class1.flatten() >= th).astype(int)
    print(f"Threshold: {th}")
    print(f"Predictions shape: {pred_flat.shape}")

    # Calcular la matriz de confusión
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()

    # Métricas:
    TPR_class1 = round(tp / (tp + fn), 3)  # True Positive Rate (Recall/Sensitivity) para clase 1 (próstata)
    print(f"TPR_class1: {TPR_class1}")
    TNR_class0 = round(tn / (tn + fp), 3)  # True Negative Rate (Specificity) para clase 0 (background)
    print(f"TNR_class0: {TNR_class0}")
    PPV_class1 = round(tp / (tp + fp), 3)  # Positive Predictive Value (Precision) para clase 1
    print(f"PPV_class1: {PPV_class1}")
    NPV_class0 = round(tn / (tn + fn), 3)  # Negative Predictive Value para clase 0
    print(f"NPV_class0: {NPV_class0}")

    # Intersection over Union (IoU) para cada clase
    iou_class0 = round(tn / (tn + fp + fn), 3)
    print(f"iou_class0: {iou_class0}")
    iou_class1 = round(tp / (tp + fp + fn), 3)
    print(f"iou_class1: {iou_class1}")

    tabla_metricas_manuales.add_data(TPR_class1, TNR_class0, PPV_class1, NPV_class0, iou_class0, iou_class1, th)

    # Guardar el IoU para este umbral
    iou_scores.append(iou_class1)

# Loguear la tabla completa al final
wandb.log({"Manual Metrics": tabla_metricas_manuales})

# Encontrar el umbral que maximiza el IoU
optimal_iou_index = np.argmax(iou_scores)
optimal_iou_threshold = thresholds[optimal_iou_index]
max_iou = iou_scores[optimal_iou_index]

print(f"Umbral óptimo basado en IoU: {optimal_iou_threshold}")
print(f"Máximo IoU obtenido: {max_iou}")

# AUROC (necesitas usar las predicciones softmax para AUROC)
true_binary_labels = all_ground_truths.flatten()
target_scores = all_softmax_predictions_class1.flatten()
print("Flattened ground truth:", gt_flat[:10])
print("Flattened predictions:", target_scores[:10])

fpr, tpr, th = roc_curve(true_binary_labels, target_scores)
auc_score = roc_auc_score(true_binary_labels, target_scores)

print(f"AUC: {auc_score}")
wandb.log({"AUC": auc_score})

# plot the ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
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

"""# Confusion matrix para obtener TP, TN, FP, FN
tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()

# Métricas:
TPR_class1 = tp / (tp + fn)  # True Positive Rate (Recall/Sensitivity) para clase 1 (próstata)
TNR_class0 = tn / (tn + fp)  # True Negative Rate (Specificity) para clase 0 (background)
PPV_class1 = tp / (tp + fp)  # Positive Predictive Value (Precision) para clase 1
NPV_class0 = tn / (tn + fn)  # Negative Predictive Value para clase 0

# Intersection over Union (IoU) para cada clase
iou_class0 = tn / (tn + fp + fn)
iou_class1 = tp / (tp + fp + fn)

# AUROC (necesitas usar las predicciones softmax para AUROC)
true_binary_labels = all_ground_truths.flatten()
target_scores = all_softmax_predictions_class1.flatten()

fpr, tpr, th = roc_curve(true_binary_labels, target_scores)
auc_score = roc_auc_score(true_binary_labels, target_scores)

# Imprimir resultados
print(f"TPR (Recall/Sensitivity) para clase 1 (próstata): {TPR_class1}")
print(f"TNR (Specificity) para clase 0 (background): {TNR_class0}")
print(f"PPV (Precision) para clase 1 (próstata): {PPV_class1}")
print(f"NPV para clase 0 (background): {NPV_class0}")
print(f"IoU para clase 0 (background): {iou_class0}")
print(f"IoU para clase 1 (próstata): {iou_class1}")
print(f"AUC: {auc_score}")

# Suponiendo que ya tienes los valores de tpr, fpr y umbrales
distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
optimal_index = np.argmin(distances)
optimal_threshold = th[optimal_index]

print(f"Umbral óptimo basado en la curva ROC: {optimal_threshold}")
print("Matriz de confusión:")
print(f"TP: {tp}, FP: {fp}")
print(f"FN: {fn}, TN: {tn}")

# Crear una tabla para almacenar las métricas
tabla_metricas_manuales = wandb.Table(columns=["TPR", "TNR", "PPV", "NPV", "IoU classe 0", "IoU classe 1", "AUC", "Umbral Optimo"])
tabla_metricas_manuales.add_data(TPR_class1, TNR_class0, PPV_class1, NPV_class0, iou_class0, iou_class1, auc_score, optimal_threshold)
wandb.log({"Manual Metrics": tabla_metricas_manuales})

# registrar la matriz de confusion como una figura
fig, ax = plt.subplots(figsize=(6, 6))
# con ploty se puede hacer un heatmap de la matriz de confusión
sns.heatmap([[tp, fp], [fn, tn]], annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticklabels(['Positive', 'Negative'])
ax.set_yticklabels(['Positive', 'Negative'])
ax.set_title('Confusion Matrix')
plt.show()
# Registrar la matriz de confusión
wandb.log({"Confusion Matrix": wandb.Image(plt)})
plt.close()

print("METRICAS DE EVALUACION ANTIGUAS")

precision = precision_score(gt_flat, pred_flat, pos_label=1, average=None)
print("Precision:", precision)

iou = jaccard_score(gt_flat, pred_flat, pos_label=1, average=None)
print("IoU (Jaccard Index):", iou)

accuracy = accuracy_score(gt_flat, pred_flat)
print("Accuracy:", accuracy)

f1 = f1_score(gt_flat, pred_flat, pos_label=1, average=None)
print("F1 Score:", f1)

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

fpr_threshold = 0.2
threshold_index = np.argmax(fpr >= fpr_threshold)
threshold = th[threshold_index]

print(f"Manual Threshold for FPR = {fpr_threshold}: {threshold}")

tabla_metricas_sklearn = wandb.Table(columns=["Precision class 0", "Precision class 1", "IoU class 0", "IoU class 1", "Accuracy", "F1 Score class 0", "F1 Score class 0", "AUC", "Threshold manual"])
tabla_metricas_sklearn.add_data(precision[0], precision[1], iou[0], iou[1], accuracy, f1[0], f1[1], auc_score, threshold)
wandb.log({"Metrics": tabla_metricas_sklearn})
"""
wandb.finish()