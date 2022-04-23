from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
def cluster_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

from sklearn.metrics import pair_confusion_matrix
def adjusted_rand_score(labels_true,labels_pred): 
    '''safer implementation of ari score calculation'''
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    tn=int(tn)
    tp=int(tp)
    fp=int(fp)
    fn=int(fn)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))

def compute_score(y_last,y):
    NMI = normalized_mutual_info_score(y, y_last)*100
    ARI = adjusted_rand_score(y, y_last)*100
    ACC = cluster_accuracy(y, y_last)*100
    return ARI,NMI,ACC
