import numpy as np
import torch


def iou_score(pred: torch.Tensor, target: torch.Tensor, n_classes: int = 5) -> (torch.Tensor, np.array):
    """
    Calculates IoU between a prediction and true label in a multi-class setting.
    :param pred: Predicted label
    :param target: True label
    :param n_classes: Number of classes

    :returns: IoU for different classes, and class occurences
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    count = np.zeros(n_classes)

    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().cpu().item()
        union = pred_inds.long().sum().cpu().item() + target_inds.long().sum().cpu().item() - intersection  # .data.cpu()[0] - intersection

        if union == 0:
            ious.append(0.0)
        else:
            count[cls] += 1
            ious.append(float(intersection) / float(max(union, 1)))

    return torch.Tensor(ious).cuda(), count
