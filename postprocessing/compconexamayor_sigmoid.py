import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
from ..preprocessing.dataset_hosp import get_train_val_test_Dataloaders  # Ensure this is the correct import
from ..models.unet3d import UNet3D
from ..config.config import IN_CHANNELS, CHECKPOINT_PATH, SPLIT_SEED, SELECTED_THRESHOLD
from ..utils.utils import resize_image_and_label
from ..training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from ..utils.plot import visualize_slices_labels_prediction_overlap, visualize_slice_predictions
from ..config.config import config
from ..utils.utils import get_largest_non_near_edge_component
import pandas as pd

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

# Initialization
all_ground_truths = []
all_argmax_predictions = []
all_predictions = []
all_softmax_predictions_class1 = []
count_predictions = 0
count_overlapping_predictions = 0

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
        
        outputs_sigmoid = torch.sigmoid(outputs)
        outputs_argmax = outputs_sigmoid.argmax(dim=1)
        
        prostate_probabilities = outputs_sigmoid.clone()
        prostate_probabilities = prostate_probabilities[0, 1, :, :, :]
        
        #change the threshold depending on the model 
        predictions = (prostate_probabilities >= SELECTED_THRESHOLD).int()

        print(f"prostate predictions shape: {predictions.shape}")

        # Apply the preprocessing to each slice
        for slice_idx in range(predictions.shape[0]): # predictions.shape[0] = 32
            predictions[slice_idx, :, :] = torch.tensor(
                #get_largest_non_edge_component(predictions[slice_idx, :, :].cpu().numpy())
                get_largest_non_near_edge_component(predictions[slice_idx, :, :].cpu().numpy()) 
            ).to(device)
          
        # Filtrado de slices sin etiquetas , POR QUE? NO ESTAMOS TESTEANDO LA RED 1, SINO LA 2 Y 3, SOLO EN LA 1
        for slice in range(resized_labels.shape[2]):
            if torch.any(resized_labels[0, 0, slice, :, :] > 0):
                all_ground_truths.append(resized_labels[0, 0, slice, :, :].cpu().numpy())
                all_predictions.append(predictions[slice, :, :].cpu().numpy())
                all_softmax_predictions_class1.append(outputs_sigmoid[0, 1, slice, :, :].cpu().numpy())
                if count_predictions < 10:
                    visualize_slice_predictions(resized_images, resized_labels, outputs, outputs_sigmoid, outputs_argmax, predictions, slice)
                    count_predictions += 1
        # Visualización para todas las slices
        if count_overlapping_predictions < 10:
            visualize_slices_labels_prediction_overlap(resized_images, resized_labels, predictions, 32, 4, figsize=(10, 60))
            count_overlapping_predictions += 1

# Convertir a numpy arrays
all_ground_truths = np.array(all_ground_truths) # all the slices of all the patients labeled and not labeled (label column in the image)
all_predictions = np.array(all_predictions) # all the predictions of all the slices of all the patients (prediction column in the image)
all_softmax_predictions_class1 = np.array(all_softmax_predictions_class1) # all the softmax predictions of all the slices of all the patients (softmax for class 1 column in the 2nd image)

# Aplanar para cálculo de métricas
gt_flat = all_ground_truths.flatten()
pred_flat = all_predictions.flatten()

# Confusion matrix para obtener TP, TN, FP, FN
tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()

# Métricas:
TPR_class1 = tp / (tp + fn)
TNR_class0 = tn / (tn + fp)
PPV_class1 = tp / (tp + fp)
NPV_class0 = tn / (tn + fn)

# Intersection over Union (IoU) para cada clase
iou_class0 = tn / (tn + fp + fn)
iou_class1 = tp / (tp + fp + fn)

# AUROC (las predicciones softmax para AUROC)
true_binary_labels = all_ground_truths.flatten()
target_scores = all_softmax_predictions_class1.flatten()
print("Flattened ground truth:", gt_flat[:10])
print("Flattened predictions:", target_scores[:10])

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

# Crear un diccionario con las métricas
metrics_dict = {
    "TPR_class1": [TPR_class1],
    "TNR_class0": [TNR_class0],
    "PPV_class1": [PPV_class1],
    "NPV_class0": [NPV_class0],
    "IoU_class0": [iou_class0],
    "IoU_class1": [iou_class1],
    "AUC": [auc_score],
    "Optimal Threshold": [optimal_threshold],
    "TP": [tp],
    "FP": [fp],
    "FN": [fn],
    "TN": [tn]
}

# Crear un DataFrame con las métricas
metrics_df = pd.DataFrame(metrics_dict)

# Guardar las métricas en un archivo Excel
metrics_filename = "output/componente_conexa_mayor_metricas.xlsx"
metrics_df.to_excel(metrics_filename, index=False)


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
confusion_matrix_filename = "output/componente_conexa_mayor_confusion_matrix.png"
fig.savefig(confusion_matrix_filename)
plt.close()