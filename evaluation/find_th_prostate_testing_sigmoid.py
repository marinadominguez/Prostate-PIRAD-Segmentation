import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..preprocessing.dataset_hosp import get_train_val_test_Dataloaders  # Ensure this is the correct import
from ..models.unet3d import UNet3D
from ..config.config import IN_CHANNELS, CHECKPOINT_PATH, SPLIT_SEED, ALPHA #, DICE_WEIGHT, FOREGROUND_WEIGHT
from ..utils.utils import resize_image_and_label
from ..training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np
import random
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from ..utils.plot import visualize_slices_labels_prediction_overlap, visualize_slice_predictions
import pandas as pd
#import segmentation_models_pytorch_3d as smp

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
# Definir el modelo con UNet++ y ResNet como encoder
"""model = smp.UnetPlusPlus(
    encoder_name="resnet34",        # ResNet-34 como encoder preentrenado
    encoder_weights="imagenet",     # Cargar pesos preentrenados en ImageNet
    in_channels=IN_CHANNELS,        # Cambia según el número de canales de tus imágenes (3 si son RGB)
    classes=1                       # Número de clases de salida, para binario usa 1
)"""

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
all_argmax_predictions = []
#all_predictions = []
all_softmax_predictions_class1 = []
count_visualize_slice_predictions = 0
count_visualize_overlap = 0

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
        print(f"Outputs shape: {outputs.shape}") # Outputs shape: torch.Size([1, 2, 32, 320, 320])
        
        # outputs_softmax = F.softmax(outputs, dim=1)
        outputs_sigmoid = torch.sigmoid(outputs)
        print(f"outputs_sigmoid shape: {outputs_sigmoid.shape}") # outputs_sigmoid shape: torch.Size([1, 2, 32, 320, 320])
        outputs_argmax = outputs_sigmoid.argmax(dim=1)
        
        prostate_probabilities = outputs_sigmoid.clone()
        prostate_probabilities = prostate_probabilities[0, 1, :, :, :]
        
        predictions = (prostate_probabilities >= 0.3).int()
        #print("Predictions shape:", predictions.shape) #Predictions shape: torch.Size([32, 320, 320])
        
        # Filtrado de slices sin etiquetas , POR QUE? NO ESTAMOS TESTEANDO LA RED 1, SINO LA 2 Y 3
        for slice in range(resized_labels.shape[2]):
            if torch.any(resized_labels[0, 0, slice, :, :] > 0):
                # Solo consideramos la slice si tiene alguna etiqueta diferente de fondo
                all_ground_truths.append(resized_labels[0, 0, slice, :, :].cpu().numpy())
                all_softmax_predictions_class1.append(outputs_sigmoid[0, 1, slice, :, :].cpu().numpy())
                
                print(f"Slice {slice} processed.") # Slice 31 processed.
                if count_visualize_slice_predictions < 10: 
                    fig = visualize_slice_predictions(resized_images, resized_labels, outputs, outputs_sigmoid, outputs_argmax, predictions, slice)
                    fig.savefig(f"output/find_opt_threshold/visualize_slice_predictions_{count_visualize_slice_predictions}.png")
                    plt.close(fig)
                    count_visualize_slice_predictions += 1
        # Visualización para todas las slices
        if count_visualize_overlap < 10:
            fig = visualize_slices_labels_prediction_overlap(resized_images, resized_labels, predictions, 32, 4, figsize=(10, 60))
            fig.savefig(f"output/find_opt_threshold/visualize_overlap_{count_visualize_overlap}.png")
            plt.close(fig)
            count_visualize_overlap += 1

# Convertir a numpy arrays
all_ground_truths = np.array(all_ground_truths) # all the slices of all the patients labeled and not labeled (label column in the image)
all_softmax_predictions_class1 = np.array(all_softmax_predictions_class1) 
gt_flat = all_ground_truths.flatten()

# Matriz de confusión
print("CALCULO PARA TODOS LOS THRESHOLDS")

# Inicializar la tabla en pd para almacenar las métricas y luego guardarlas en una tabla de excel
tablas_metricas_manuales = pd.DataFrame(columns=["TPR", "TNR", "PPV", "NPV", "IoU class 0", "IoU class 1", "Threshold"])

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

    # Guardar las métricas en la tabla
    tablas_metricas_manuales = tablas_metricas_manuales.append({"TPR": TPR_class1, "TNR": TNR_class0, "PPV": PPV_class1, "NPV": NPV_class0, "IoU class 0": iou_class0, "IoU class 1": iou_class1, "Threshold": th}, ignore_index=True)

    # Guardar el IoU para este umbral
    iou_scores.append(iou_class1)

# Loguear la tabla completa
tablas_metricas_manuales.to_excel("output/find_opt_threshold/tablas_metricas_manuales.xlsx")

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
plt.savefig("output/find_opt_threshold/ROC_curve.png")
plt.close()
print("ROC curve plotted and saved successfully! Check the output folder: output/find_opt_threshold/ROC_curve.png")