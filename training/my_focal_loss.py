import torch 
import torch.nn.functional as F
from focal_loss import FocalLoss

class CustomFocalLoss(FocalLoss):
    def __init__(self, gamma=2, alpha=0.25, reduction="mean", axis=1, loss_mode="binary"):
        """
        Custom Focal Loss implementation that uses reshape instead of view
        """
        super(CustomFocalLoss, self).__init__(gamma=gamma, reduction=reduction)
        self.axis = axis
        self.loss_mode = loss_mode
        self.alpha = alpha


    def forward(self, input, target):
        """
        Override the forward method to use reshape instead of view
        """
        input_soft = torch.softmax(input, dim=self.axis)

        # Create a mask depending on the loss mode
        if self.loss_mode == "binary":
            mask = target * self.alpha + (1 - target) * (1 - self.alpha)
        elif self.loss_mode == "multiclass":
            mask = F.one_hot(target, num_classes=input.shape[1]) * self.alpha
            mask = torch.sum(mask, dim=1)

        # Replace view with reshape
        mask = mask.reshape(-1)
        input_soft = input_soft.reshape(-1)
        target = target.reshape(-1)
        
        # Calculate the focal loss as usual
        loss = F.binary_cross_entropy(input_soft, target, reduction="none")
        loss = loss * (1 - input_soft) ** self.gamma
        loss = loss * mask

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss

class CustomFocalLoss4finetuning(FocalLoss):
    def __init__(self, gamma=2, alpha=0.25, reduction="mean", axis=1, loss_mode="binary"):
        """
        Custom Focal Loss implementation that is compatible with autocast
        """
        super(CustomFocalLoss4finetuning, self).__init__(gamma=gamma, reduction=reduction)
        self.axis = axis
        self.loss_mode = loss_mode
        self.alpha = alpha

    def forward(self, input, target):
        # Do not apply softmax; use logits directly for BCE with logits
        # Create a mask depending on the loss mode
        if self.loss_mode == "binary":
            mask = target * self.alpha + (1 - target) * (1 - self.alpha)
        elif self.loss_mode == "multiclass":
            mask = F.one_hot(target, num_classes=input.shape[1]) * self.alpha
            mask = torch.sum(mask, dim=1)

        # Reshape tensors to match binary_cross_entropy_with_logits requirements
        mask = mask.reshape(-1)
        target = target.reshape(-1).float()  # Ensure target is float for BCE with logits
        input = input.reshape(-1)

        # Calculate the focal loss with binary_cross_entropy_with_logits
        loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        pt = torch.exp(-loss)  # pt = probability if sigmoid were applied
        focal_loss = (1 - pt) ** self.gamma * loss * mask

        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss


class CustomFocalForegroundDiceLoss(FocalLoss):
    def __init__(self, gamma=2, alpha=0.25, reduction="mean", axis=1, loss_mode="binary", dice_weight=0.5, foreground_weight=0.7):
        """
        Custom Focal + Dice Loss with added focus on foreground class
        """
        super(CustomFocalForegroundDiceLoss, self).__init__(gamma=gamma, reduction=reduction)
        self.axis = axis
        self.loss_mode = loss_mode
        self.alpha = alpha
        self.dice_weight = dice_weight
        self.foreground_weight = foreground_weight  # Weight for foreground Dice Loss

    def dice_loss(self, pred, target, smooth=1e-5):
        """
        Dice Loss implementation
        """
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice_score

    def forward(self, input, target):
        input_soft = torch.softmax(input, dim=self.axis)

        # Mask for binary or multiclass
        if self.loss_mode == "binary":
            mask = target * self.alpha + (1 - target) * (1 - self.alpha)
        elif self.loss_mode == "multiclass":
            mask = F.one_hot(target, num_classes=input.shape[1]) * self.alpha
            mask = torch.sum(mask, dim=1)

        mask = mask.reshape(-1)
        input_soft = input_soft.reshape(-1)
        target = target.reshape(-1)
        
        # Focal Loss
        focal_loss = F.binary_cross_entropy(input_soft, target, reduction="none")
        focal_loss = focal_loss * (1 - input_soft) ** self.gamma
        focal_loss = focal_loss * mask

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == "sum":
            focal_loss = torch.sum(focal_loss)

        # Foreground Dice Loss
        foreground_mask = (target == 1).float()
        dice_foreground_loss = self.dice_loss(input_soft, foreground_mask)

        # Combine Focal Loss and Foreground Dice Loss
        total_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * (self.foreground_weight * dice_foreground_loss)
        return total_loss


class CustomFocalDiceLoss(FocalLoss):
    def __init__(self, gamma=2, alpha=0.25, reduction="mean", axis=1, loss_mode="binary", dice_weight=0.5):
        """
        Custom Focal + Dice Loss implementation that uses reshape instead of view
        """
        super(CustomFocalDiceLoss, self).__init__(gamma=gamma, reduction=reduction)
        self.axis = axis
        self.loss_mode = loss_mode
        self.alpha = alpha
        self.dice_weight = dice_weight  # Weight for the Dice Loss

    def dice_loss(self, input, target, smooth=1e-5):
        """
        Dice Loss implementation
        """
        input_flat = input.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (input_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice_score  # Dice loss is 1 - Dice coefficient

    def forward(self, input, target):
        """
        Override the forward method to use reshape instead of view
        """
        input_soft = torch.softmax(input, dim=self.axis)

        # Create a mask depending on the loss mode
        if self.loss_mode == "binary":
            mask = target * self.alpha + (1 - target) * (1 - self.alpha)
        elif self.loss_mode == "multiclass":
            mask = F.one_hot(target, num_classes=input.shape[1]) * self.alpha
            mask = torch.sum(mask, dim=1)

        # Replace view with reshape
        mask = mask.reshape(-1)
        input_soft = input_soft.reshape(-1)
        target = target.reshape(-1)
        
        # Calculate Focal Loss
        focal_loss = F.binary_cross_entropy(input_soft, target, reduction="none")
        focal_loss = focal_loss * (1 - input_soft) ** self.gamma
        focal_loss = focal_loss * mask

        # Calculate Dice Loss
        dice_loss = self.dice_loss(input_soft, target)
        
        # Combine Focal Loss and Dice Loss
        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == "sum":
            focal_loss = torch.sum(focal_loss)

        total_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        return total_loss

