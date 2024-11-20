import matplotlib.pyplot as plt
import numpy as np
import pydicom
import wandb

import matplotlib.pyplot as plt
import numpy as np
import wandb

def visualize_slices_labels_prediction_overlap(resized_images, resized_labels, predictions, num_rows, num_cols, figsize=(10, 60)):
    """
    Visualizes a grid of slices, labels, and overlaid predictions.

    Parameters:
    - resized_images: Tensor containing the input images. (size = [batch_size, channels, slices, height, width])
    - resized_labels: Tensor containing the ground truth labels. (size = [batch_size, channels, slices, height, width])
    - outputs: Tensor containing the predicted outputs. (size = [slices, height, width])
    - num_rows: Number of rows in the grid (default is 28).
    - num_cols: Number of columns in the grid (default is 3).
    - figsize: Size of the figure for plotting (default is (10, 60)).
    """
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i in range(num_rows):
        # Extract the image, label, and prediction for the current slice
        image = resized_images[0, 0, i, :, :].cpu().numpy()
        label = resized_labels[0, 0, i, :, :].cpu().numpy()
        #prediction = outputs_argmax[0, i, :, :].cpu().numpy()
        prediction = predictions[i, :, :].cpu().numpy() 

        # Display the image
        ax[i, 0].imshow(image, cmap='gray')
        ax[i, 0].axis('off')
        ax[i, 0].set_title('Slice ' + str(i))

        # Display the label
        ax[i, 1].imshow(label, cmap='gray')
        ax[i, 1].axis('off')
        ax[i, 1].set_title('Label ' + str(i))

        # Display the prediction 
        ax[i, 2].imshow(prediction, cmap='gray')
        ax[i, 2].axis('off')
        ax[i, 2].set_title('Prediction ' + str(i))

        # Display the prediction overlaid in the label overlaid on the image 
        ax[i, 3].imshow(image, cmap='gray')
        ax[i, 3].imshow(label, cmap='jet', alpha=0.5)
        ax[i, 3].imshow(prediction, cmap='jet', alpha=0.5)
        ax[i, 3].axis('off')
        ax[i, 3].set_title('Overlap ' + str(i))

    plt.tight_layout()
    # Save and log the plot
    generated_plot = plt.gcf()
    wandb.log({"Slices Labels Prediction Overlap": wandb.Image(generated_plot)})
    plt.close()


def visualize_slice_predictions(resized_images, resized_labels, outputs, outputs_softmax, outputs_argmax, thresholded, slice_index, patient_index=None, figsize=(35, 5)):
    """
    Visualizes the predictions for a specific slice.

    Parameters:
    - resized_images: Tensor containing the input images.
    - resized_labels: Tensor containing the ground truth labels.
    - outputs: Tensor containing the predicted outputs.
    - outputs_softmax: Tensor containing the softmax probabilities.
    - outputs_argmax: Tensor containing the argmax of the predictions.
    - prostate_probabilities: Tensor containing the thresholded probabilities.
    - slice_index: Index of the slice to visualize.
    - patient_index: Optional index of the patient (for logging purposes).
    - figsize: Size of the figure for plotting (default is (35, 5)).
    """
    plt.figure(figsize=figsize)

    # Input Image
    plt.subplot(1, 8, 1)
    plt.imshow(resized_images[0, 0, slice_index, :, :].cpu().numpy(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    # Ground Truth
    plt.subplot(1, 8, 2)
    plt.imshow(resized_labels[0, 0, slice_index, :, :].cpu().numpy(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    # Predicted for class 0
    plt.subplot(1, 8, 3)
    plt.imshow(outputs[0, 0, slice_index, :, :].cpu().numpy(), cmap='gray')
    plt.title(f"Predicted for class 0")
    plt.axis('off')

    # Predicted for class 1
    plt.subplot(1, 8, 4)
    plt.imshow(outputs[0, 1, slice_index, :, :].cpu().numpy(), cmap='gray')
    plt.title(f"Predicted for class 1")
    plt.axis('off')

    # Softmax for class 0
    plt.subplot(1, 8, 5)
    plt.imshow(outputs_softmax[0, 0, slice_index, :, :].cpu().numpy(), cmap='gray')
    plt.title('Softmax for class 0')
    plt.axis('off')

    # Softmax for class 1
    plt.subplot(1, 8, 6)
    plt.imshow(outputs_softmax[0, 1, slice_index, :, :].cpu().numpy(), cmap='gray')
    plt.title('Softmax for class 1')
    plt.axis('off')

    # Argmax
    plt.subplot(1, 8, 7)
    plt.imshow(outputs_argmax[0, slice_index, :, :].cpu().numpy(), cmap='gray')
    plt.title('Argmax')
    plt.axis('off')

    # Thresholded
    plt.subplot(1, 8, 8)
    plt.imshow(thresholded[slice_index, :, :].cpu().numpy(), cmap='gray')
    plt.title('Thresholded')
    plt.axis('off')

    # Save and log the plot
    generated_plot = plt.gcf()
    wandb.log({"Test Patients Results": wandb.Image(generated_plot)})
    if patient_index is not None:
        print(f"Patient {patient_index} slice {slice_index} plotted")

    plt.close()


def visualize_crf_results(resized_images, resized_labels, outputs_argmax, prostate_probability_map, refined_prediction, slice_idx):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    # Convert the tensors to CPU and then to numpy for visualization, if they are PyTorch tensors
    image_slice = resized_images[0, 0, slice_idx, :, :].cpu().numpy()
    label_slice = resized_labels[0, 0, slice_idx, :, :].cpu().numpy()
    argmax_slice = outputs_argmax[0, slice_idx, :, :].cpu().numpy()
    print(prostate_probability_map.shape)
    #breakpoint()
    prostate_prob_map_slice = prostate_probability_map[slice_idx, :, :].cpu().numpy()
    refined_pred_slice = refined_prediction[slice_idx, :, :].cpu().numpy()
    
    axs[0].imshow(image_slice, cmap='gray')
    axs[0].set_title('Original Image Slice')

    axs[1].imshow(label_slice, cmap='gray')
    axs[1].set_title('Ground Truth Label')

    axs[2].imshow(argmax_slice, cmap='gray')
    axs[2].set_title('Model Argmax Prediction')

    axs[3].imshow(prostate_prob_map_slice, cmap='hot', vmin=0, vmax=1)
    axs[3].set_title('Prostate Probability Map')

    axs[4].imshow(refined_pred_slice, cmap='gray')
    axs[4].set_title('CRF Refined Prediction')

    for ax in axs:
        ax.axis('off')

    plt.show()
    wandb.log({"Refined Prediction": wandb.Image(plt)})
    plt.close()
        