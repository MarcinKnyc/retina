import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the accuracy score."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return (y_true == y_pred).mean()


def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the sensitivity score."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return ((y_true == 255) & (y_pred == 255)).sum() / (y_true == 255).sum()


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the specificity score."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return ((y_true == 0) & (y_pred == 0)).sum() / (y_true == 0).sum()


def get_quality(masks: list, postprocessed_masks: list) -> tuple:
    """Computes the quality metrics for the predictions."""
    accuracy_scores = [accuracy_score(mask, postprocessed_mask) for mask, postprocessed_mask in zip(masks, postprocessed_masks)]
    sensitivity_scores = [sensitivity_score(mask, postprocessed_mask) for mask, postprocessed_mask in zip(masks, postprocessed_masks)]
    specificity_scores = [specificity_score(mask, postprocessed_mask) for mask, postprocessed_mask in zip(masks, postprocessed_masks)]
    return accuracy_scores, sensitivity_scores, specificity_scores

