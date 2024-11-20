import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from HOSP_pipeline.preprocessing.dataset_hosp import get_train_val_test_Dataloaders  # Ensure this is the correct import
from HOSP_pipeline.models.unet3d import UNet3D
from HOSP_pipeline.config.config import IN_CHANNELS, CHECKPOINT_PATH, WANDB_TEST_NAME
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
import segmentation_models_pytorch_3d as smp

wandb.init(project="hops_pipeline", entity="marina-dominguez", name="OptThresh_Unet++_ResNet34_CombinedLoss_Interpolation_Preprocess")

# Log the entire config dictionary
wandb.config.update(config)

# Now you can access it like this:
print("Dataset path:", wandb.config.dataset_path)
print("Number of epochs:", wandb.config.training_epoch)

# Ensure NUM_CLASSES matches the pretrained model's configuration
PRETRAINED_NUM_CLASSES = 2  # Set to 2 equal to the number of classes in the trained model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL
# model = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)
model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # ResNet-34 como encoder preentrenado
    encoder_weights="imagenet",     # Cargar pesos preentrenados en ImageNet
    in_channels=IN_CHANNELS,        # Cambia según el número de canales de tus imágenes (3 si son RGB)
    classes=1                       # Número de clases de salida, para binario usa 1
)

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
        prostate_probabilities = prostate_probabilities[0, 0, :, :, :]
        
        # WITH THE CHOSEN THRESHOLD FROM TESTING_FOR_ALL_THRESHOLDS
        predictions = (prostate_probabilities >= 0.6134).int()
        # 0.6613 is the chosen threshold for Unet++ resnet34 with combined loss
        # pre processing interpolation from scratch, threshold = 0.4409
        # pre processing with interpolation fine-tuned, threshold = 0.01847
        # lower threshold for the model trained from scratch, threshold = 0.2021
        # increase threshold for the model trained from scratch, threshold = 0.6134
        #print("Predictions shape:", predictions.shape) #Predictions shape: torch.Size([32, 320, 320])
        
        for slice in range(resized_labels.shape[2]):
            if torch.any(resized_labels[0, 0, slice, :, :] > 0): # Solo consideramos la slice si tiene alguna etiqueta diferente de fondo
                all_ground_truths.append(resized_labels[0, 0, slice, :, :].cpu().numpy())
                #all_argmax_predictions.append(outputs_argmax[0, slice, :, :].cpu().numpy())
                all_predictions.append(predictions[slice, :, :].cpu().numpy())
                all_softmax_predictions_class1.append(outputs_sigmoid[0, 0, slice, :, :].cpu().numpy())
                print(f"Slice {slice} processed.") # Slice 31 processed.
            #print("Image shape:", resized_images[0, 0, slice, :, :].shape) # Image shape: torch.Size([320, 320])
            #print("Ground truth shape:", resized_labels[0, 0, slice, :, :].shape) # Ground truth shape: torch.Size([320, 320])
            #print("Output shape:", outputs[0, :, slice, :, :].shape) # Output shape: torch.Size([2, 320, 320])
            #print("Prostate probabilities shape:", outputs_softmax[0, 1, slice, :, :].shape) # Prostate probabilities shape: torch.Size([320, 320])
            #print("Prediction shape:", outputs_argmax[0, slice, :, :].shape) # Prediction shape: torch.Size([320, 320])
            #print("Prediction shape:", predictions.shape) # Prediction shape: torch.Size([32, 320, 320])
            # Solo consideramos la slice si tiene alguna etiqueta diferente de fondo
            
            #if torch.any(resized_labels[0, 0, slice, :, :] > 0):
                visualize_slice_predictions(resized_images, resized_labels, outputs, outputs_sigmoid, outputs_argmax, predictions, slice)
        # Visualización para todas las slices
        # arguments : resized_images, resized_labels, outputs, num_rows, num_cols, figsize=(10, 60)
        print("Visualizing all slices")
        #print("Image shape:", resized_images[0, 0, :, :, :].shape) # Image shape: torch.Size([32, 320, 320])
        #print("Ground truth shape:", resized_labels[0, 0, :, :, :].shape) # Ground truth shape: torch.Size([32, 320, 320])
        #print("Output shape:", outputs_argmax[0, :, :, :].shape) # Output shape: torch.Size([32, 320, 320])
        # print("predictions shape: ", predictions.shape) # predictions shape:  torch.Size([32, 320, 320])
        visualize_slices_labels_prediction_overlap(resized_images, resized_labels, predictions, 32, 4, figsize=(10, 60))

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

wandb.finish()