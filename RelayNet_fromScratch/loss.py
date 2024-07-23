import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMultiClassLogisticLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedMultiClassLogisticLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        targets = targets.squeeze(1).long()  # Ensure targets have correct shape
        loss = F.nll_loss(log_probs, targets, weight=self.class_weights)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
        targets = targets.squeeze(1).long()
        
        # Convert targets to one-hot encoding
        one_hot_targets = torch.eye(num_classes)[targets].permute(0, 3, 1, 2).float().to(logits.device)
        
        m1 = probs.view(probs.size(0), num_classes, -1)
        m2 = one_hot_targets.view(one_hot_targets.size(0), num_classes, -1)

        # Debugging output
        print(f"probs shape: {probs.shape}")
        print(f"one_hot_targets shape: {one_hot_targets.shape}")
        print(f"m1 shape: {m1.shape}")
        print(f"m2 shape: {m2.shape}")

        intersection = (m1 * m2).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (m1.sum(dim=2) + m2.sum(dim=2) + self.smooth)
        return 1 - dice.mean()


# Example weight scheme
class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0, 3.0, 1.0, 1.2, 2.5, 1.0])  # Adjust as necessary
