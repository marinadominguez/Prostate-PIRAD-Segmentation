import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

def apply_crf(image, probs, prostate_probability_map=None, num_classes=2):
    """
    Apply CRF to refine predictions based on image, softmax probabilities, and a spatial prior.
    
    Args:
    - image: The original image slice (grayscale with shape [height, width]).
    - probs: Softmax probabilities output by the model (with shape [num_classes, height, width]).
    - prostate_probability_map: A prior probability map derived from statistical analysis (optional).
    - num_classes: Number of classes (e.g., 2 for binary segmentation).

    Returns:
    - Refined segmentation after CRF post-processing.
    """
    h, w = image.shape[:2]
    
    # Initialize DenseCRF
    d = dcrf.DenseCRF2D(w, h, num_classes)
    
    # Prepare unary potentials from the softmax probabilities
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)
    
    # Adjust pairwise Gaussian potentials (for smoothness)
    d.addPairwiseGaussian(sxy=(3, 3), compat=1)  # Lower compat for less aggressive smoothing
    
    # Expand dimensions of the grayscale image to include a channel dimension
    image_3d = np.expand_dims(image, axis=-1)  # Shape: (height, width, 1)
    
    # Adjust pairwise bilateral potentials (for intensity/color-based smoothing)
    pairwise_bilateral = create_pairwise_bilateral(sdims=(10, 10), schan=(5,), img=image_3d, chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=3)  # Lower compat for less aggressive smoothing
    
    # Perform inference
    Q = d.inference(10)  # Try more iterations for better convergence
    refined_prediction = np.argmax(Q, axis=0).reshape((h, w))
    
    return refined_prediction
