import numpy as np
from ._roc import compute_roc

def micro_auroc(labels, preds):
    roc_micro = np.zeros(np.size(preds, 1))
    cl_i = 0
    for i in range(len(roc_micro)):
        unseen_preds, unseen_labels = preds[:, i], labels[:, i]
        auroc = compute_roc(unseen_labels, unseen_preds)
        roc_micro[cl_i] = auroc
        cl_i += 1
    return roc_micro