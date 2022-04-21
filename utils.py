import torch
import numpy as np


def metrics(prediction, target):

    prediction_binary = torch.ge(prediction, 0.5).float()
    N = target.numel()

    # True positives, true negative, false positives, false negatives calculation
    tp = torch.nonzero(prediction_binary * target).shape[0]
    tn = torch.nonzero((1 - prediction_binary) * (1 - target)).shape[0]
    fp = torch.nonzero(prediction_binary * (1 - target)).shape[0]
    fn = torch.nonzero((1 - prediction_binary) * target).shape[0]

    # Metrics
    accuracy = (tp + tn) / N
    precision = 0. if tp == 0 else tp / (tp + fp)
    recall = 0. if tp == 0 else tp / (tp + fn)
    specificity = 0. if tn == 0 else tn / (tn + fp)
    f1 = 0. if precision == 0 or recall == 0 else (2 * precision * recall) / (precision + recall)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity}
