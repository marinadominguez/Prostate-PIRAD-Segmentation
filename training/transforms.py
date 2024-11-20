from monai.transforms import (
    Compose,
    ScaleIntensityd,  # For normalization
    ToTensord,  # To convert arrays to tensors
    Resized,  
)

import numpy as np

#Transforms to be applied on training instances
train_transform = Compose([
    # If resizing is necessary to ensure spatial dimension compatibility
    # Resized(keys=['image', 'label'], spatial_size=(128, 128, 128), mode=('trilinear', 'nearest')),
    ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),  # Normalize images only
    ToTensord(keys=['image', 'label']),
])

#Cuda version of "train_transform"
train_transform_cuda = Compose([
    # If resizing is necessary to ensure spatial dimension compatibility
    # Resized(keys=['image', 'label'], spatial_size=(128, 128, 128), mode=('trilinear', 'nearest')),
    ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),  # Normalize images only
    ToTensord(keys=['image', 'label'], device='cuda'),
])

#Transforms to be applied on validation instances
val_transform = Compose(
    [   
        # If resizing is necessary to ensure spatial dimension compatibility
        # Resized(keys=['image', 'label'], spatial_size=(128, 128, 128), mode=('trilinear', 'nearest')),
        ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),  # Normalize images only
        ToTensord(keys=['image', 'label']),
    ]
)

#Cuda version of "val_transform"
val_transform_cuda = Compose(
    [   
        # If resizing is necessary to ensure spatial dimension compatibility
        # Resized(keys=['image', 'label'], spatial_size=(128, 128, 128), mode=('trilinear', 'nearest')),
        ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),  # Normalize images only
        ToTensord(keys=['image', 'label'], device='cuda')
    ]
)