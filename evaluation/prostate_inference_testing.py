import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..preprocessing.dataset_hosp import get_train_val_test_Dataloaders  # Ensure this is the correct import
from ..models.unet3d import UNet3D
from ..config.config import NUM_CLASSES, IN_CHANNELS, BACKGROUND_AS_CLASS, CHECKPOINT_PATH, WANDB_TEST_NAME
from ..utils.utils import resize_image_and_label, plot_patient_slices, plot_confusion_matrix
from ..training.transforms import (train_transform, train_transform_cuda, val_transform, val_transform_cuda)
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from matplotlib.colors import LogNorm
import torch.nn.functional as F

wandb.init(project="hops_pipeline", entity="marina-dominguez", name="Pretrained-95Epoch-Threshold-FPR0.15-plot-0.4")

# Ensure NUM_CLASSES matches the pretrained model's configuration
PRETRAINED_NUM_CLASSES = 2  # Set to 2 based on the error message

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL
model = UNet3D(in_channels=IN_CHANNELS, num_classes=PRETRAINED_NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
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
        
        for slice in range(resized_labels.shape[2]):
            plt.figure(figsize=(35, 5))

            plt.subplot(1, 8, 1)
            plt.imshow(resized_images[0, 0, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title('Input Image')
            plt.axis('off')

            plt.subplot(1, 8, 2)
            plt.imshow(resized_labels[0, 0, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            plt.subplot(1, 8, 3)
            plt.imshow(outputs[0, 0, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title(f"Predicted for class 0")
            plt.axis('off')

            plt.subplot(1, 8, 4)
            plt.imshow(outputs[0, 1, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title(f"Predicted for class 1")
            plt.axis('off')

            plt.subplot(1, 8, 5)
            plt.imshow(outputs_softmax[0, 0, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title('Softmax for class 0')
            plt.axis('off')

            plt.subplot(1, 8, 6)
            plt.imshow(outputs_softmax[0, 1, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title('Softmax for class 1')
            plt.axis('off')

            plt.subplot(1, 8, 7)
            plt.imshow(outputs_argmax[0, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title('Argmax')
            plt.axis('off')
            
            plt.subplot(1, 8, 8)
            plt.imshow(prostate_probabilities[slice, :, :].cpu().numpy(), cmap='gray')
            plt.title('Thresholded')
            plt.axis('off')

            generated_plot = plt.gcf()
            wandb.log({"Test Patients Results": wandb.Image(generated_plot)})
            print(f"Patient {i} slice {slice} plotted")

            plt.close()
        
        all_ground_truths.append(resized_labels.cpu().numpy())
        all_argmax_predictions.append(outputs_argmax.cpu().numpy())
        all_softmax_predictions_class1.append(outputs_softmax[0, 1, :, :, :].cpu().numpy())

all_ground_truths = np.vstack(all_ground_truths)
all_argmax_predictions = np.vstack(all_argmax_predictions)
all_softmax_predictions_class1 = np.vstack(all_softmax_predictions_class1)

gt_flat = all_ground_truths.flatten()
pred_flat = all_argmax_predictions.flatten()

precision = precision_score(gt_flat, pred_flat, pos_label=1, average='weighted')
print("Precision:", precision)

iou = jaccard_score(gt_flat, pred_flat, pos_label=1, average='weighted')
print("IoU (Jaccard Index):", iou)

accuracy = accuracy_score(gt_flat, pred_flat)
print("Accuracy:", accuracy)

f1 = f1_score(gt_flat, pred_flat, pos_label=1, average='weighted')
print("F1 Score:", f1)

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
