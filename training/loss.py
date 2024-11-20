import torch
import torch.nn as nn
import torch.nn.functional as F
from HOSP_pipeline.config.config import BCE_WEIGHTS, NUM_CLASSES, BACKGROUND_AS_CLASS, DICE_WEIGHT, ALPHA_TVERSKY, BETA_TVERSKY, GAMMA
from HOSP_pipeline.utils.utils import to_one_hot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if BACKGROUND_AS_CLASS: NUM_CLASSES += 1


def dice_loss(pred, target, num_classes=NUM_CLASSES):
    # pred shape: torch.Size([1, 5, 32, 320, 320])
    # target shape: torch.Size([1, 32, 320, 320])
    
    #print("pred shape for loss: ", pred.shape)
    #print("target shape for loss: ", target.shape)
    # pred shape for loss:  torch.Size([1, 5, 32, 320, 320])
    # target shape for loss:  torch.Size([1, 1, 32, 320, 320])
    target_for_loss = target.squeeze(1)  # Remove channel dimension from target

    pred = F.softmax(pred, dim=1)  # Apply softmax to convert to probabilities
    # pred shape after softmax: torch.Size([1, 5, 32, 320, 320])
    
    # print("pred shape after softmax: ", pred.shape)
    # pred shape after softmax:  torch.Size([1, 5, 32, 320, 320])

    target_one_hot = F.one_hot(target_for_loss, num_classes=num_classes)  # This adds a new last dimension for classes
    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # Reorder to match [batch, classes, depth, height, width]
    # target_one_hot shape: torch.Size([1, 5, 32, 320, 320])
    
    dice_loss = 0
    for class_index in range(num_classes):  # Include background class (class 0)
        # Class: 0, 1, 2, 3, 4
        pred_class = pred[:, class_index, :, :, :] # pred_class shape: torch.Size([1, 32, 320, 320])
        target_class = target_one_hot[:, class_index, :, :, :] # target_class shape: torch.Size([1, 32, 320, 320])
        
        intersection = (pred_class * target_class).sum()
        total = pred_class.sum() + target_class.sum()
        class_dice_loss = 1 - (2 * intersection + 1e-6) / (total + 1e-6)
        dice_loss += class_dice_loss
    
    return dice_loss / num_classes  # Average including the background

class DiceLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        return dice_loss(inputs, targets, self.num_classes)
    
class DiceWCELoss(nn.Module):
    def __init__(self, weight=None):
        """
        Inicializa la función de pérdida combinada.
        :param weight: Tensor de pesos para las clases para la pérdida de entropía cruzada ponderada.
        Estos pesos deben reflejar la importancia relativa de las clases y su frecuencia inversa en los datos.
        """
        super(DiceWCELoss, self).__init__()
        self.weight = weight
        self.wce = nn.CrossEntropyLoss(weight) if weight is not None else nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        """
        Calcula la pérdida combinada como la suma de la pérdida de entropía cruzada ponderada y la pérdida de Dice.
        :param inputs: Predicciones del modelo.
        :param targets: Etiquetas verdaderas.
        :return: Pérdida combinada.
        """
        # print("inputs shape for loss: ", inputs.shape)
        # print("targets shape for loss: ", targets.shape)
        # inputs shape for loss:  torch.Size([1, 5, 32, 320, 320])
        # targets shape for loss:  torch.Size([1, 1, 32, 320, 320])

        targets_for_loss = targets.squeeze(1)  # Remove channel dimension from targets
        # print("targets_for_loss shape: ", targets_for_loss.shape)
        # targets_for_loss shape:  torch.Size([1, 32, 320, 320])

        wce_loss = self.wce(inputs, targets_for_loss)
        #print("wce_loss: ", wce_loss)
        dice_loss = self.dice(inputs, targets)
        #print("dice_loss: ", dice_loss)

        combined_loss = wce_loss + dice_loss
        print("combined_loss: ", combined_loss)

        return combined_loss


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dice_weight=None):
        super(DiceCELoss, self).__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        #self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(BCE_WEIGHTS).float().to(device))
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        # Entropía cruzada
        ce_loss = self.cross_entropy(y_pred, y_true)
        # inputs shape for loss:  torch.Size([1, 5, 32, 320, 320])
        # targets shape for loss:  torch.Size([1, 1, 32, 320, 320])
        # Pérdida de Dice
        dice = dice_loss(y_pred, y_true, self.num_classes)
        # Combinar las pérdidas
        return ce_loss + self.dice_weight * dice


def jaccard_index_multiclass(y_true, y_pred, num_classes):
    """Calcula el índice de Jaccard para múltiples clases."""
    # Aplicar softmax para convertir logits a probabilidades
    y_pred = F.softmax(y_pred, dim=1)
    jac = 0.0
    # Calcular Jaccard para cada clase
    for c in range(num_classes):
        true = (y_true == c).float()
        pred = y_pred[:, c]
        intersection = (true * pred).sum()
        union = true.sum() + pred.sum() - intersection
        # Evitar división por cero
        jac += (intersection + 1e-7) / (union + 1e-7)
    # Promedio de Jaccard sobre todas las clases
    jac /= num_classes
    return jac

class JaccardMulticlassLoss(nn.Module):
    def __init__(self, num_classes):
        super(JaccardMulticlassLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        # Índice de Jaccard
        jac = jaccard_index_multiclass(y_true, y_pred, self.num_classes)
        return jac

class JaccardCELoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(JaccardCELoss, self).__init__()
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(BCE_WEIGHTS).float().to(device))

    def forward(self, y_pred, y_true):
        # Entropía cruzada
        ce_loss = self.cross_entropy(y_pred, y_true)
        # Índice de Jaccard
        jac = jaccard_index_multiclass(y_true, y_pred, self.num_classes)
        # Combinar las pérdidas
        return ce_loss - jac

class JaccardWCELoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(JaccardWCELoss, self).__init__()
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(BCE_WEIGHTS).float().to(device))

    def forward(self, y_pred, y_true):
        # Entropía cruzada
        ce_loss = self.cross_entropy(y_pred, y_true)
        # Índice de Jaccard
        jac = jaccard_index_multiclass(y_true, y_pred, self.num_classes)
        # Combinar las pérdidas
        return ce_loss + jac


# from https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
    
#PyTorch
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA_TVERSKY, beta=BETA_TVERSKY, num_classes=NUM_CLASSES):
        #comment out if your model contains a sigmoid or equivalent activation layer
        print("inputs shape for loss: ", inputs.shape) # torch.Size([1, 5, 32, 320, 320])
        #inputs = F.sigmoid(inputs) 
        #inputs = F.softmax(inputs, dim=1)
        print("targets shape for loss: ", targets.shape) # torch.Size([1, 1, 32, 320, 320])
        
        # Asegurándose de que inputs esté en forma de probabilidades y targets en one-hot
        # inputs debe pasar por una softmax si aún no es una probabilidad
        # print("maximum value of the inputs: ", inputs.max()) # 3.3841
        # print("minimum value of the inputs: ", inputs.min()) # -2.7912
        inputs = F.softmax(inputs, dim=1) 
        # For each pixel, the softmax function ensures that the probabilities across all channels 
        # (classes) sum up to 1, making it easier to determine which class is the most probable for a given pixel.

        # print("inputs shape after softmax: ", inputs.shape) # torch.Size([1, 5, 32, 320, 320])

        #print("maximum value of the inputs after softmax: ", inputs.max()) # 1.0
        #print("minimum value of the inputs after softmax: ", inputs.min()) # 0.0  

        targets_one_hot = to_one_hot(targets, num_classes)
        print("targets_one_hot shape: ", targets_one_hot.shape)
                
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        print("inputs shape after flatten: ", inputs.shape)
        targets = targets_one_hot.view(-1)
        print("targets shape after flatten: ", targets.shape)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA_TVERSKY, beta=BETA_TVERSKY, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Ensure alpha is a scalar or a tensor with a size that matches the number of classes
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Make sure that the inputs are probabilities
        inputs = F.softmax(inputs, dim=1) # [1, 5, 32, 320, 320]  
        
        # Squeeze the channel dimension of targets, if necessary
        targets = targets.squeeze(1).long() # [1, 32, 320, 320] 

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).to(inputs.device) 
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float() # [1, 5, 32, 320, 320]

        # Calculate logpt
        logpt = F.log_softmax(inputs, dim=1) # [1, 5, 32, 320, 320]
        logpt = (logpt * targets_one_hot).sum(dim=1) # [1, 32, 320, 320]

        # Calculate focal loss
        pt = logpt.exp() # [1, 32, 320, 320]
        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * logpt

        return focal_loss.mean()

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Ensure the targets are in one-hot format
        one_hot_targets = torch.zeros_like(inputs).scatter_(1, targets, 1)

        # Calculate BCE loss using logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, one_hot_targets, reduction='none')
        
        # Calculate probabilities using sigmoid (only for calculating pt)
        probabilities = torch.sigmoid(inputs)
        
        # Correct calculation of pt
        pt = torch.where(one_hot_targets == 1, probabilities, 1 - probabilities)
        
        # Calculate the focal loss components
        alpha_t = self.alpha * one_hot_targets + (1 - self.alpha) * (1 - one_hot_targets)
        modulating_factor = (1.0 - pt) ** self.gamma
        
        # Apply the focal modulation to BCE loss
        focal_loss = alpha_t * modulating_factor * bce_loss

        return focal_loss.mean()  # return the mean of the focal loss
        

    
    """

    def forward(self, inputs, targets):
        # Ensure inputs are probabilities (assuming inputs are logits)
        inputs = F.softmax(inputs, dim=1) # [1, 5, 32, 320, 320]

        # Squeeze the single-channel dimension from ground_truth and ensure it's a long tensor for one_hot
        targets = targets.squeeze(1).long() # [1, 32, 320, 320]

        # One-hot encode targets to match the shape of inputs
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 4, 1, 2, 3).float()
        # targets_one_hot shape: torch.Size([1, 5, 32, 320, 320])
        targets_one_hot = targets_one_hot.to(inputs.device)

        # Compute the log probabilities
        logpt = F.log_softmax(inputs, dim=1) # [1, 5, 32, 320, 320]
        # Gather the probabilities of the correct class
        logpt = torch.gather(logpt, 1, targets.unsqueeze(1)) # [1, 1, 32, 320, 320]
        logpt = logpt.squeeze(1) # [1, 32, 320, 320]
        pt = torch.exp(logpt) # [1, 32, 320, 320]

        # Compute focal loss
        alpha = self.alpha 
        loss = -alpha * ((1 - pt) ** self.gamma) * logpt # [1, 32, 320, 320]

        return loss.mean()
        """

"""
    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1) # [1, 5, 32, 320, 320]
        # Convert targets to one-hot encoding
        targets = targets.squeeze(1).long() # [1, 32, 320, 320]
        print("targets shape after squeeze: ", targets.shape)
        print("one hot targets shape: ", F.one_hot(targets.long(), num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float().shape)
        targets_one_hot = F.one_hot(targets.long(), num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
        targets_one_hot = targets_one_hot.to(inputs.device)

        # Calculate log probabilities
        logpt = -F.cross_entropy(inputs, targets_one_hot, reduction='none')
        pt = torch.exp(logpt)

        # Calculate Focal Loss
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt *= at

        loss = -((1-pt)**self.gamma) * logpt

        return loss.mean()
"""