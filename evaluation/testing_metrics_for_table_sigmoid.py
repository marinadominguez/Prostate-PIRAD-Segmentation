import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..preprocessing.dataset_hosp import get_train_val_test_Dataloaders  # Ensure this is the correct import
from ..models.unet3d import UNet3D
from ..config.config import IN_CHANNELS, CHECKPOINT_PATH, SELECTED_THRESHOLD
from ..utils.utils import resize_image_and_label
from ..training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import pandas as pd

# Ensure NUM_CLASSES matches the pretrained model's configuration
PRETRAINED_NUM_CLASSES = 2  # Set to 2 equal to the number of classes in the trained model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
all_predictions = []
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
        
        print(f"Final input shape to the model: {resized_images.shape}")
        outputs = model(resized_images)
        # outputs_softmax = F.softmax(outputs, dim=1)
        outputs_sigmoid = torch.sigmoid(outputs)
        outputs_argmax = outputs_sigmoid.argmax(dim=1)
        
        prostate_probabilities = outputs_sigmoid.clone()
        prostate_probabilities = prostate_probabilities[0, 1, :, :, :]
        
        # WITH THE CHOSEN THRESHOLD FROM TESTING_FOR_ALL_THRESHOLDS
        predictions = (prostate_probabilities >= SELECTED_THRESHOLD).int()
        
        for slice in range(resized_labels.shape[2]):
            if torch.any(resized_labels[0, 0, slice, :, :] > 0): # Solo consideramos la slice si tiene alguna etiqueta diferente de fondo
                all_ground_truths.append(resized_labels[0, 0, slice, :, :].cpu().numpy())
                #all_argmax_predictions.append(outputs_argmax[0, slice, :, :].cpu().numpy())
                all_predictions.append(predictions[slice, :, :].cpu().numpy())
                all_softmax_predictions_class1.append(outputs_sigmoid[0, 1, slice, :, :].cpu().numpy())
                print(f"Slice {slice} processed.") 

# Convertir a numpy arrays
all_ground_truths = np.array(all_ground_truths) # all the slices of all the patients labeled and not labeled (label column in the image)
all_predictions = np.array(all_predictions) # all the predictions of all the slices of all the patients (prediction column in the image)
all_softmax_predictions_class1 = np.array(all_softmax_predictions_class1) # all the softmax predictions of all the slices of all the patients (softmax for class 1 column in the 2nd image)

# Aplanar para cálculo de métricas
gt_flat = all_ground_truths.flatten()
pred_flat = all_predictions.flatten()
   
print("METRICAS DE EVALUACION NUEVAS")
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

# Crear una tabla para almacenar las métricas y guardarla en excel 
tabla_metricas_manuales = pd.DataFrame(columns=["TPR class 1", "TNR class 0", "PPV class 1", "NPV class 0", "IoU class 0", "IoU class 1", "AUC", "Threshold manual"])
tabla_metricas_manuales = tabla_metricas_manuales.append({"TPR class 1": TPR_class1, "TNR class 0": TNR_class0, "PPV class 1": PPV_class1, "NPV class 0": NPV_class0, "IoU class 0": iou_class0, "IoU class 1": iou_class1, "AUC": auc_score, "Threshold manual": optimal_threshold}, ignore_index=True)
tabla_metricas_manuales.to_excel("output/tabla_metricas_optimal_threshold.xlsx", index=False)

print("Tabla de métricas guardada en output/tabla_metricas_optimal_threshold.xlsx")