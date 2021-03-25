import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class CrossEntropyLossAux(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True, reduction = 'mean'):
        super(CrossEntropyLossAux, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce, reduction = reduction)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2*0.4