import numpy as np
import sklearn.metrics as metrics


def get_quality(masks: list, postprocessed_masks: list) -> dict:
    """Computes the quality metrics for the predictions."""
    y_true = np.concatenate(masks).ravel()
    y_pred = np.concatenate(postprocessed_masks).ravel()

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    tnr = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    fnr = fn / (fn + tp)
    ppv = tp / (tp + fp)
    fdr = fp / (fp + tp)

    fpr, tpr, _ = metrics.roc_curve(
        np.concatenate(masks).ravel(),
        np.concatenate(postprocessed_masks).ravel(),
        pos_label=255)

    auc = metrics.auc(fpr, tpr)

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
        tpr_roc=tpr,
        fpr_roc=fpr,
    )
