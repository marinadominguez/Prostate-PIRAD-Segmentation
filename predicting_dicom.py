from fastai.basics import *
from fastai.vision import models
from fastai.vision.all import *
from fastai.metrics import *
from fastai.data.all import *
from fastai.callback import *

# SemTorch
from semtorch import get_segmentation_learner
import os, random, torch,cv2
from pathlib import Path
torch.cuda.set_device(0)
import numpy as np
import PIL
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import sys
import pydicom

number_of_the_seed = 41
random.seed(number_of_the_seed)
set_seed(number_of_the_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import PIL
from PIL import Image, ImageOps
import torchvision.transforms as transforms

#Dicom path
dicom_path = sys.argv[1]
path_model="models/hrnet.pth"

#Read dicom image
dicom = pydicom.dcmread(dicom_path)
image = dicom.pixel_array
pixel_min = image.min()
pixel_max = image.max()
image = ((image - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = torch.jit.load(path_model)
model = model.cpu()


mean= Normalize.from_stats(*imagenet_stats).mean.flatten()
std= Normalize.from_stats(*imagenet_stats).std.flatten()
def transform_image(image):
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
    image_aux = image
    return my_transforms(image_aux).unsqueeze(0).to(device)

def prediction(image=image, aux=model):
  img = Image.fromarray(image).convert('RGB')
  image = transforms.Resize((480,480))(img)
  tensor = transform_image(image=image)
  aux.to(device)
  with torch.no_grad():
      outputs = aux(tensor)
  outputs = torch.argmax(outputs,1)
  mask = np.array(outputs.cpu())
  mask=np.reshape(mask,(480,480))
  return np.array(transforms.Resize(img.shape)(Image.fromarray(mask.astype('uint8'))))

mask = prediction()
num_labels, labels = cv2.connectedComponents(mask)
mask=(labels==1).astype('uint8')
cv2.imwrite('output.png', 255*mask)
print('This patient has cancer:' + str(np.sum(mask) > 0))