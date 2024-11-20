import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

"""
def resize_image_and_label(image, label, target_size=(32, 320, 320)): # og size: (20, 256, 256)
    if image.dim() == 5:
        print("Image is 5D.")
        resized_image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=True)
        if label.shape[1] == 20:
            print("20 slices in ground truth. Resizing...")
            label = label.unsqueeze(1)  # Add a channel dimension
        resized_label = F.interpolate(label.float(), size=target_size, mode='nearest').long()
    
    elif image.dim() == 6: 
        print("Image is 6D!! Resizing...")
        image = image.view(image.size(0), image.size(1), -1, image.size(4), image.size(5))
        resized_image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=True)
        if label.dim() == 4:
            label = label.unsqueeze(1)  # Add a channel dimension
        resized_label = F.interpolate(label.float(), size=target_size, mode='nearest').long()
    return resized_image, resized_label
"""

def resize_image_and_label(image, label, target_size=(32, 320, 320)):
    # Assume image and label are 5D with shapes [batch_size, channels, depth, height, width]
    print("Resizing image and label.")

    # Ensure the image and label have a channel dimension if they're missing
    if image.dim() == 4:
        print("Image is 4D.")
        print("Image shape:", image.shape)
        image = image.unsqueeze(0)  # Add a channel dimension -> (1, 1, D, H, W)
        print("Image shape after adding channel dimension:", image.shape)
    if label.dim() == 4:
        print("Label is 4D.")
        print("Label shape:", label.shape)
        label = label.unsqueeze(0)  # Add a channel dimension -> (1, 1, D, H, W)
        print("Label shape after adding channel dimension:", label.shape)

    # Resize image using trilinear interpolation
    resized_image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=True)
    
    # Note: Assuming label is initially [batch_size, 1, depth, height, width], and we temporarily adjust it for F.interpolate
    # label = label.squeeze(1)  # Temporarily remove channel dimension for interpolation
    resized_label = F.interpolate(label, size=target_size, mode='nearest')  # Add dummy channel dimension for interpolation
    #resized_label = resized_label.squeeze(1).long()  # Remove the dummy channel dimension and convert back to long
    # print("Label shape:", label.shape) # ([1, 5, 20, 256, 256])
    #resized_label = F.interpolate(label.float(), size=target_size, mode='nearest').long()
    """
    Resized image shape: torch.Size([1, 1, 32, 320, 320])
    label shape: torch.Size([1, 1, 32, 320, 320])
    """
    print("Resized image shape:", resized_image.shape)
    print("Resized label shape:", resized_label.shape)

    return resized_image, resized_label

def resize_prostate_presence_map(prostate_presence_map, target_size=(32, 320, 320)):
    # prostate_presence_maps is a 2d numpy array
    print("Resizing prostate presence map.")
    # we can not use the F.interpolate function with numpy arrays, so we need to convert the numpy array to a tensor
    prostate_presence_map = prostate_presence_map.to(dtype=torch.float32)
    print("Prostate presence map type:", prostate_presence_map.dtype)
    print("Prostate presence map shape:", prostate_presence_map.shape)
    # Ensure the prostate presence map has a channel dimension if it's missing
    if prostate_presence_map.dim() == 2: 
        print("Prostate presence map is 2D.")
        prostate_presence_map = prostate_presence_map.unsqueeze(0).unsqueeze(0).unsqueeze(0) # Add a channel dimension -> (1, 1, D, H, W)
        print("Prostate presence map shape after adding channel dimension:", prostate_presence_map.shape)
    # Resize prostate presence map using trilinear interpolation
    resized_prostate_presence_map = F.interpolate(prostate_presence_map, size=target_size, mode='trilinear', align_corners=True)
    print("Resized prostate presence map shape:", resized_prostate_presence_map.shape)

    return resized_prostate_presence_map

def resize_image(image, target_size=(32, 320, 320)):
    # Image is 5D with shapes [batch_size, channels, depth, height, width]
    print("Resizing image.")
    # Ensure the image has a channel dimension if it's missing
    if image.dim() == 4:
        print("Image is 4D.")
        print("Image shape:", image.shape)
        image = image.unsqueeze(0)
        print("Image shape after adding channel dimension:", image.shape)
    # Resize image using trilinear interpolation
    resized_image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=True)
    print("Resized image shape:", resized_image.shape)
    return resized_image

def to_one_hot(targets, n_classes):
    batch_size, *dims = targets.size() # [batch_size, channels, slices, height, width]
    one_hot_targets = torch.zeros((batch_size, n_classes) + tuple(dims[1:]), device=targets.device) # [batch_size, n_classes, slices, height, width]
    one_hot_targets = one_hot_targets.scatter_(1, targets, 1) # here we are using the scatter_ method to fill the tensor with the one-hot encoding
    return one_hot_targets

# visualize the og img, the ground truth and the prediction
def plot_patient_slices(image, ground_truth, outputs):

    for slice in range(32):
        if outputs.argmax(dim=1)[0, slice, :, :].sum() > 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(image[0, 0, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title('Input Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(ground_truth[0, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(outputs.argmax(dim=1)[0, slice, :, :].cpu().numpy(), cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
            
            generated_plot = plt.gcf() # get current figure

            plt.close()

    return generated_plot
    
def plot_confusion_matrix(all_ground_truths, all_predictions):
    # Calculate confusion matrix
    cm = confusion_matrix(all_ground_truths, all_predictions)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot using seaborn for a nicer layout
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    generated_cm = plt.gcf() # get current figure
    plt.close()
    return generated_cm

def total_confusion(correct_labels, predicted_labels, num_classes):
    """
    Construye una matriz de confusión para clasificación multietiqueta.

    Args:
    - correct_labels: Etiquetas verdaderas.
    - predicted_labels: Etiquetas predichas.
    - num_classes: Número total de clases.

    Returns:
    - Matriz de confusión (num_classes x num_classes).
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(correct_labels, predicted_labels):
        for c in range(num_classes):
            if true[c] == 1:
                if pred[c] == 1:
                    cm[c, c] += 1  # True Positive
                else:
                    for i in range(num_classes):
                        if pred[i] == 1:
                            cm[c, i] += 1  # False Negative (labeled as c, predicted as i)
            else:
                if pred[c] == 1:
                    for i in range(num_classes):
                        if true[i] == 1:
                            cm[i, c] += 1  # False Positive (labeled as i, predicted as c)

    # Los verdaderos negativos no son relevantes en cada entrada de la matriz de confusión para multietiqueta
    # ya que estamos interesados en las predicciones hechas, no en las no hechas.
    return cm

def segmentation_to_class_presence(segmentation, num_classes):
    """
    Convierte un mapa de segmentación en un vector de presencia de clase.
    Args:
        segmentation (np.array): El mapa de segmentación, donde cada elemento es el índice de clase de un píxel.
        num_classes (int): El número total de clases, incluyendo el fondo.

    Returns:
        np.array: Un vector binario de longitud num_classes donde cada elemento indica la presencia (1) o ausencia (0) de una clase.
    """
    classes_present = np.zeros(num_classes, dtype=int)
    unique_classes = np.unique(segmentation)
    classes_present[unique_classes] = 1
    return classes_present

def segmentation_to_max_class_presence(segmentation, num_classes):
    max_class_present = np.zeros(num_classes, dtype=int) # max_class_present = [0, 0, 0, 0, 0]
    #print("Max class present:", max_class_present)
    unique_classes = np.unique(segmentation) # unique_classes = [0, 1, 2, 3, 4]
    # print("Unique classes:", unique_classes)
    # from the unique classes, we get the maximum class
    max_class = max(unique_classes) # max_class = 4
    # print("Max class:", max_class)
    max_class_present[max_class] = 1 # max_class_present = [0, 0, 0, 0, 1]
    # print("Max class present:", max_class_present)
    return max_class_present


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