import numpy as np
import sklearn.metrics as metrics
from .models_and_predictions import apply_threshold


def get_quality(masks: list, postprocessed_masks: list, threshold) -> dict:
    """Computes the quality metrics for the predictions."""
    y_true = np.concatenate(masks).ravel()
    y_pred = np.concatenate(postprocessed_masks).ravel()

    y_true_threshold = apply_threshold(y_true, threshold)
    y_pred_threshold = apply_threshold(y_pred, threshold)

    tn, fp, fn, tp = metrics.confusion_matrix(
        y_true_threshold, y_pred_threshold).ravel()

    tnr = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    fnr = fn / (fn + tp)
    ppv = tp / (tp + fp)
    fdr = fp / (fp + tp)

    fpr, tpr, _ = metrics.roc_curve(
        y_true_threshold, y_pred_threshold, pos_label=255)
    fpr_roc, tpr_roc, _ = metrics.roc_curve(y_true, y_pred, pos_label=255)

    auc = metrics.auc(fpr_roc, tpr_roc)

    return dict(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        tpr=np.float32(tpr[1]),
        fpr=np.float32(fpr[1]),
        tnr=tnr,
        acc=acc,
        fnr=fnr,
        ppv=ppv,
        fdr=fdr,
        auc=np.float32(auc),
        tpr_roc=tpr_roc,
        fpr_roc=fpr_roc,
    )
