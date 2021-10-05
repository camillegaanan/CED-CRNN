"""
Tool to metrics calculation through data and label (string and string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import string
import unicodedata
import editdistance
import numpy as np
from numpy.lib.function_base import average
from sklearn.metrics import precision_score, recall_score, f1_score


def ocr_metrics(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer, ser, acc = [], [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        pd, gt = pd.lower(), gt.lower()

        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode("ASCII")
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode("ASCII")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_cer, gt_cer = list(pd), list(gt)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.split(), gt.split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        # acc.append(accuracy_score(gt_wer, pd_wer))

        pd_ser, gt_ser = [pd], [gt]
        dist = editdistance.eval(pd_ser, gt_ser)
        ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    f1_macro = f1_score(ground_truth, predicts, average = 'macro')
    f1_micro = f1_score(ground_truth, predicts, average = 'micro')
    f1_weighted = f1_score(ground_truth, predicts, average = 'weighted')

    precision_macro = precision_score(ground_truth, predicts, average = 'macro')
    precision_micro = precision_score(ground_truth, predicts, average = 'micro')
    precision_weighted = precision_score(ground_truth, predicts, average = 'weighted')

    recall_macro = recall_score(ground_truth, predicts, average = 'macro')
    recall_micro = recall_score(ground_truth, predicts, average = 'micro')
    recall_weighted = recall_score(ground_truth, predicts, average = 'weighted')

    metrics = [cer, wer, ser]
    metrics = np.mean(metrics, axis=1)
    metrics = np.append(metrics,precision_macro)
    metrics = np.append(metrics,precision_micro)
    metrics = np.append(metrics,precision_weighted)
    metrics = np.append(metrics,recall_macro)
    metrics = np.append(metrics,recall_micro)
    metrics = np.append(metrics,recall_weighted)
    metrics = np.append(metrics,f1_macro)
    metrics = np.append(metrics,f1_micro)
    metrics = np.append(metrics,f1_weighted)

    return metrics
