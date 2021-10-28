"""
Tool to metrics calculation through data and label (string and string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import math
import string
import unicodedata
import editdistance
import numpy as np
# from numpy.lib.function_base import average
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from scipy.stats import shapiro
# from scipy.stats import ttest_rel, wilcoxon
import csv


def ocr_metrics(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer = [], []
    num_of_recognized_words =  0
    num_of_expected_words = 0
    acc = 0
    word_dict = {}

    for (pd, gt) in zip(predicts, ground_truth):
        # pd, gt = pd.lower(), gt.lower()

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

        for word in pd_wer:
            if word in gt_wer:
                acc+=1

        for word in gt_wer:
            if word not in pd_wer:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1

        num_of_expected_words += len(gt_wer)
        num_of_recognized_words += len(pd_wer)

        
    word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)

    with open("misclassified_words.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["word", "count"])
        for i in word_dict:
            writer.writerow([i[0], i[1]])
    

    accuracy = (acc/num_of_expected_words)*100
    recall = (acc/num_of_recognized_words)*100
    precision = (acc/num_of_expected_words)*100
    f1 = 2*((precision*recall) / (precision+recall))

    metrics = [cer, wer]
    metrics = np.mean(metrics, axis=1)
    metrics = np.append(metrics,accuracy)
    metrics = np.append(metrics,recall)
    metrics = np.append(metrics,precision)
    metrics = np.append(metrics,f1)
    # metrics = np.append(metrics,precision_macro)
    # metrics = np.append(metrics,precision_micro)
    # metrics = np.append(metrics,precision_weighted)
    # metrics = np.append(metrics,recall_macro)
    # metrics = np.append(metrics,recall_micro)
    # metrics = np.append(metrics,recall_weighted)
    # metrics = np.append(metrics,f1_macro)
    # metrics = np.append(metrics,f1_micro)                                                                                                                                                          
    # metrics = np.append(metrics,f1_weighted)

    return word_dict, cer, metrics


# ground = ['Prednisolone: 5-60 mg per day qds', 'Dobutamine: 2.5-15 mcg/kg/min', 'Tramadol: 50-100 mg as needed every 4 to 6 hours', 'Azathioprine: 3-5 mg/kg Per os OD', 'Dobutamine: 2.5-15 mcg/kg/min']
# pred = ['Cetrimaone: 0-5 mg Per s qas','Cetrimaone: 0-5 mg Per s qas','Cetrimaone: 0-5 mg Per s qas','Cetrimaone: 0-5 mg Per s qas','Cetrimaone: 0-5 mg Per s qas']


# _,cer, m =ocr_metrics(predicts=pred, ground_truth=ground)
# print(m[2])
# print(m[3])
# print(m[4])
# print(m[5])

# cer_fajardo = [0.42424242424242425, 0.7931034482758621, 0.6875, 0.48484848484848486, 0.7931034482758621, 0.7021276595744681, 0.6785714285714286, 0.7931034482758621, 0.5357142857142857, 0.48484848484848486, 0.5357142857142857, 0.4166666666666667, 0.5151515151515151, 0.5510204081632653, 0.7021276595744681, 0.7021276595744681, 0.4166666666666667, 0.6944444444444444, 0.4166666666666667, 0.5357142857142857, 0.5510204081632653, 0.6875, 0.7021276595744681, 0.7142857142857143, 0.7142857142857143, 0.5510204081632653, 0.6875, 0.48484848484848486, 0.6944444444444444, 0.5357142857142857, 0.42424242424242425, 0.7931034482758621, 0.6785714285714286, 0.4166666666666667, 0.5625, 0.6785714285714286, 0.6875, 0.7931034482758621, 0.6785714285714286, 0.7021276595744681, 0.42424242424242425, 0.7142857142857143, 0.5357142857142857, 0.6785714285714286, 0.6785714285714286, 0.6875, 0.6875, 0.6944444444444444, 0.42424242424242425, 0.6785714285714286, 0.6875, 0.6785714285714286, 0.7142857142857143, 0.5510204081632653, 0.4166666666666667, 0.7931034482758621, 0.6875, 0.48484848484848486, 0.42424242424242425, 0.5625, 0.6944444444444444, 0.6875, 0.6944444444444444, 0.5357142857142857, 0.7021276595744681, 0.7142857142857143, 0.5510204081632653, 0.5151515151515151, 0.7931034482758621, 0.4166666666666667, 0.48484848484848486, 0.5510204081632653, 0.6944444444444444, 0.5357142857142857, 0.7931034482758621, 0.42424242424242425, 0.5625, 0.5625, 0.6785714285714286, 0.6944444444444444, 0.42424242424242425, 0.5625, 0.42424242424242425, 0.5510204081632653, 0.5357142857142857, 0.5625, 0.48484848484848486, 0.5357142857142857, 0.6944444444444444, 0.4166666666666667, 0.4166666666666667, 0.6785714285714286, 0.5625, 0.4166666666666667, 0.5714285714285714, 0.42424242424242425, 0.42424242424242425, 0.5510204081632653, 0.6944444444444444, 0.42424242424242425, 0.6875, 0.42424242424242425, 0.7142857142857143, 0.6785714285714286, 0.7021276595744681, 0.5625, 0.5625, 0.7142857142857143, 0.7142857142857143, 0.7021276595744681, 0.7931034482758621, 0.6944444444444444, 0.7931034482758621, 0.6944444444444444, 0.5357142857142857, 0.53125, 0.7142857142857143, 0.6875, 0.7931034482758621, 0.48484848484848486, 0.6944444444444444, 0.6875, 0.5625, 0.4166666666666667, 0.7021276595744681, 0.7142857142857143, 0.4166666666666667, 0.48484848484848486, 0.48484848484848486, 0.5357142857142857, 0.7142857142857143, 0.6785714285714286, 0.5625, 0.7142857142857143, 0.48484848484848486, 0.4166666666666667, 0.7931034482758621, 0.4166666666666667, 0.5357142857142857, 0.7021276595744681, 0.5357142857142857, 0.7021276595744681, 0.42424242424242425, 0.5510204081632653, 0.5510204081632653, 0.7021276595744681, 0.6875, 0.5510204081632653]
# cer_simpler = [0.30303030303030304, 0.0, 0.034482758620689655, 0.0, 0.25, 0.0, 0.06896551724137931, 0.0, 0.0, 0.5757575757575758, 0.0, 0.0, 0.0, 0.020833333333333332, 0.6595744680851063, 0.041666666666666664, 0.0, 0.17142857142857143, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.02127659574468085, 0.0, 0.14705882352941177, 0.0, 0.07692307692307693, 0.0, 0.0, 0.02857142857142857, 0.0, 0.0, 0.0, 0.027777777777777776, 0.034482758620689655, 0.027777777777777776, 0.0, 0.0, 0.0, 0.6774193548387096, 0.0, 0.02127659574468085, 0.034482758620689655, 0.030303030303030304, 0.0, 0.04081632653061224, 0.16666666666666666, 0.0, 0.0, 0.0, 0.0, 0.034482758620689655, 0.25, 0.0, 0.03125, 0.06060606060606061, 0.48936170212765956, 0.0, 0.03125, 0.5555555555555556, 0.034482758620689655, 0.041666666666666664, 0.0, 0.0, 0.05555555555555555, 0.027777777777777776, 0.0, 0.3548387096774194, 0.0, 0.5897435897435898, 0.0, 0.08163265306122448, 0.0, 0.0, 0.0, 0.20512820512820512, 0.6785714285714286, 0.029411764705882353, 0.0, 0.0, 0.20588235294117646, 0.0, 0.55, 0.32, 0.02040816326530612, 0.0, 0.0, 0.0, 0.0, 0.59375, 0.46938775510204084, 0.0, 0.0, 0.0, 0.16326530612244897, 0.02702702702702703, 0.0, 0.0, 0.027777777777777776, 0.0, 0.7714285714285715, 0.0, 0.0, 0.0, 0.125, 0.05714285714285714, 0.0, 0.0, 0.020833333333333332, 0.03571428571428571, 0.0, 0.09375, 0.0, 0.0, 0.06060606060606061, 0.08333333333333333, 0.0, 0.8064516129032258, 0.19148936170212766, 0.6857142857142857, 0.0, 0.06666666666666667, 0.0, 0.5882352941176471, 0.10344827586206896, 0.06896551724137931, 0.0, 0.061224489795918366, 0.020833333333333332, 0.0, 0.027777777777777776, 0.0, 0.12121212121212122, 0.0, 0.3191489361702128, 0.03125, 0.0, 0.10416666666666667, 0.14705882352941177, 0.0, 0.0, 0.2777777777777778, 0.05555555555555555, 0.0, 0.0, 0.14285714285714285]

# cer_diff = []
# for x, y in zip(cer_fajardo, cer_simpler):
#     cer_diff.append(x-y)

# shapiro_statistic, p = shapiro(cer_diff)
# print('Shapiro statistic of the differences in CER:', shapiro_statistic)
# print('p-value of the differences in CER:', p)

# r = pyasl.generalizedESD(cer_diff, 5, 0.05, fullOutput=True)
# print('Number of outliers for the differences in CER:',r[0])

# shapiro_statistic, p = shapiro(cer_fajardo)
# print('Shapiro statistic of the CER of basis:', shapiro_statistic)
# print('p-value of the CER of basis:', p)

# r = pyasl.generalizedESD(cer_fajardo, 5, 0.05, fullOutput=True)
# print('Number of outliers for the CER of basis:',r[0])

# shapiro_statistic, p = shapiro(cer_simpler)
# print('Shapiro statistic of the CER of proposed model:', shapiro_statistic)
# print('p-value of the CER of proposed model:', p)

# r = pyasl.generalizedESD(cer_simpler, 5, 0.05, fullOutput=True)
# print('Number of outliers for the CER of proposed model:',r[0])

# # r = ttest_rel(cer_fajardo, cer_simpler)
# # print(r)

# r = wilcoxon(cer_diff)
# print(r)

# r = wilcoxon(cer_diff, alternative="less")
# print(r)

# r = wilcoxon(cer_diff, alternative="greater")
# print(r)