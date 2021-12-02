import string
import unicodedata
import editdistance
import numpy as np
import os
from scipy.stats import shapiro
from scipy.stats import wilcoxon
from PyAstronomy import pyasl
import csv


def ocr_metrics(predicts, ground_truth, output_path, norm_accentuation=False, norm_punctuation=False):

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer = [], []
    num_of_recognized_words =  0
    num_of_expected_words = 0
    acc_word = 0
    acc = 0
    word_dict = {}
    word_dict2 = {}
    acc_label_1 = 0
    acc_label_2 = 0
    acc_label_3 = 0
    acc_label_4 = 0
    acc_label_5 = 0
    acc_label_6 = 0
    acc_label_7 = 0
    acc_label_8 = 0
    acc_label_9 = 0
    acc_label_10 = 0
    acc_label_11 = 0
    acc_label_12 = 0
    acc_word_label = [0,0,0,0,0,0,0,0,0,0,0,0]
    num_of_recognized_words_label =  [0,0,0,0,0,0,0,0,0,0,0,0]
    num_of_expected_words_label = [0,0,0,0,0,0,0,0,0,0,0,0]
    cer_label= [0,0,0,0,0,0,0,0,0,0,0,0]
    wer_label= [0,0,0,0,0,0,0,0,0,0,0,0]
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

        for word in pd_wer:
            if word in gt_wer:
                acc_word+=1

        for word in gt_wer:
            if word not in pd_wer:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
            else:
                if word in word_dict2:
                    word_dict2[word] += 1
                else:
                    word_dict2[word] = 1

        num_of_expected_words += len(gt_wer)
        num_of_recognized_words += len(pd_wer)

        if gt.find("azathioprine") != -1:
            i = 0
        elif gt.find("ceftriaxone") != -1:
            i = 1
        elif gt.find("chlorpromazine") != -1:
            i = 2
        elif gt.find("dobutamine") != -1:
            i = 3
        elif gt.find("hydroxyzine") != -1:   
            i = 4
        elif gt.find("lorazepam") != -1:
            i = 5
        elif gt.find("metronidazole") != -1:
            i = 6
        if gt.find("prednisolone") != -1:
            i = 7
        elif gt.find("quinine") != -1:
            i = 8
        elif gt.find("risperidone") != -1:
            i = 9
        elif gt.find("rituximab") != -1:
            i = 10
        elif gt.find("tramadol") != -1:
            i = 11
            
        for word in pd_wer:
                if word in gt_wer:
                    acc_word_label[i]+=1
        num_of_expected_words_label[i] += len(gt_wer)
        num_of_recognized_words_label[i] += len(pd_wer)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer_label[i] += dist / (max(len(pd_cer), len(gt_cer)))
        dist = editdistance.eval(pd_wer, gt_wer)
        wer_label[i] += dist / (max(len(pd_wer), len(gt_wer)))
        
        if pd == gt:
            acc+=1
            if gt.find("azathioprine") != -1:
                acc_label_1+=1
            elif gt.find("ceftriaxone") != -1:
                acc_label_2+=1
            elif gt.find("chlorpromazine") != -1:
                acc_label_3+=1
            elif gt.find("dobutamine") != -1:
                acc_label_4+=1
            elif gt.find("hydroxyzine") != -1:   
                acc_label_5+=1
            elif gt.find("lorazepam") != -1:
                acc_label_6+=1
            elif gt.find("metronidazole") != -1:
                acc_label_7+=1
            elif gt.find("prednisolone") != -1:
                acc_label_8+=1
            elif gt.find("quinine") != -1:
                acc_label_9+=1
            elif gt.find("risperidone") != -1:
                acc_label_10+=1
            elif gt.find("rituximab") != -1:
                acc_label_11+=1
            elif gt.find("tramadol") != -1:
                acc_label_12+=1
            
    acc_label = [acc_label_1,acc_label_2,acc_label_3,acc_label_4,acc_label_5,acc_label_6,acc_label_7,acc_label_8,acc_label_9,acc_label_10,acc_label_11,acc_label_12]

    word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    word_dict2 = sorted(word_dict2.items(), key=lambda x: x[1], reverse=True)

    with open(os.path.join(output_path, "misclassified_words.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["word", "count"])
        for i in word_dict:
            writer.writerow([i[0], i[1]])

    with open(os.path.join(output_path, "classified_words.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["word", "count"])
        for i in word_dict2:
            writer.writerow([i[0], i[1]])
    
    accuracy = (acc/len(ground_truth))*100
    recall = (acc_word/num_of_recognized_words)*100
    precision = (acc_word/num_of_expected_words)*100
    f1 = 2*((precision*recall) / (precision+recall))

    recall_label=[0,0,0,0,0,0,0,0,0,0,0,0]
    precision_label=[0,0,0,0,0,0,0,0,0,0,0,0]
    f1_label=[0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(12):
        recall_label[i] = (acc_word_label[i]/num_of_recognized_words_label[i])*100
        precision_label[i] = (acc_word_label[i]/num_of_expected_words_label[i])*100
        f1_label[i] = 2*((precision_label[i]*recall_label[i]) / (precision_label[i]+recall_label[i]))
        if (i == 6 or i == 8 or i == 9 or i == 11):
            cer_label[i] = cer_label[i]/13
            wer_label[i] = wer_label[i]/13
        else:
            cer_label[i] = cer_label[i]/12
            wer_label[i] = wer_label[i]/12

    labels = [recall_label,precision_label,f1_label,cer_label,wer_label]

    metrics = [cer, wer]
    metrics = np.mean(metrics, axis=1)
    metrics = np.append(metrics,accuracy)
    metrics = np.append(metrics,recall)
    metrics = np.append(metrics,precision)
    metrics = np.append(metrics,f1)
    
    return cer, metrics, acc_label, labels

def statistical_test():
    #fajardo no noise ced
    cer_fajardo = [0.3055555555555556, 0.0, 0.0, 0.02127659574468085, 0.0, 0.41379310344827586, 0.16326530612244897, 0.0, 0.0, 0.04081632653061224, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02040816326530612, 0.0, 0.0, 0.0, 0.0, 0.0, 0.030303030303030304, 0.08333333333333333, 0.05555555555555555, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4827586206896552, 0.0, 0.10416666666666667, 0.0, 0.5, 0.0, 0.0, 0.02857142857142857, 0.0, 0.0, 0.0, 0.0, 0.4166666666666667, 0.0, 0.5454545454545454, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.08163265306122448, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.030303030303030304, 0.0, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.3783783783783784, 0.0, 0.45454545454545453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7428571428571429, 0.04081632653061224, 0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.02127659574468085, 0.027777777777777776, 0.15625, 0.04, 0.0, 0.06896551724137931, 0.0, 0.0, 0.0, 0.0, 0.0, 0.375, 0.0, 0.05405405405405406, 0.0, 0.0, 0.21052631578947367, 0.0, 0.0, 0.0, 0.0, 0.631578947368421, 0.0, 0.0, 0.0, 0.0, 0.10204081632653061, 0.0, 0.0, 0.0, 0.027777777777777776, 0.027777777777777776, 0.0, 0.08333333333333333, 0.0, 0.0, 0.07692307692307693, 0.08333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45714285714285713, 0.0, 0.40625, 0.0, 0.0, 0.22448979591836735]

    #cabais no noise skeleton
    cer_simpler = [0.08333333333333333, 0.0, 0.0, 0.06, 0.0, 0.0, 0.6530612244897959, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14, 0.0, 0.0, 0.027777777777777776, 0.0, 0.0, 0.06060606060606061, 0.0, 0.0, 0.0, 0.0, 0.16326530612244897, 0.0, 0.0, 0.10638297872340426, 0.0, 0.0, 0.034482758620689655, 0.0, 0.2916666666666667, 0.0, 0.22, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.48484848484848486, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12121212121212122, 0.0, 0.0, 0.0, 0.13725490196078433, 0.0, 0.02857142857142857, 0.2857142857142857, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03125, 0.0, 0.0, 0.0, 0.0, 0.15151515151515152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6857142857142857, 0.02040816326530612, 0.0, 0.0, 0.0, 0.0, 0.4230769230769231, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1111111111111111, 0.0, 0.0, 0.030303030303030304, 0.0, 0.0, 0.0, 0.0, 0.35294117647058826, 0.0, 0.0, 0.0, 0.02702702702702703, 0.04081632653061224, 0.0, 0.0, 0.0, 0.05263157894736842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.36363636363636365, 0.0, 0.5625, 0.0, 0.0, 0.0]

    #fajardo no noise skeleton
    # cer_fajardo = [0.0, 0.0, 0.0, 0.061224489795918366, 0.0, 0.06666666666666667, 0.6122448979591837, 0.0, 0.0, 0.02040816326530612, 0.0, 0.0, 0.041666666666666664, 0.0, 0.18181818181818182, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04081632653061224, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30303030303030304, 0.0, 0.02857142857142857, 0.0, 0.10526315789473684, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.41379310344827586, 0.0, 0.020833333333333332, 0.06060606060606061, 0.24489795918367346, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2727272727272727, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3541666666666667, 0.0, 0.0, 0.1836734693877551, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.030303030303030304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6571428571428571, 0.0, 0.0, 0.0, 0.0, 0.0, 0.041666666666666664, 0.0, 0.0, 0.15625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1388888888888889, 0.0, 0.0, 0.2571428571428571, 0.0, 0.027777777777777776, 0.0, 0.0, 0.030303030303030304, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.027777777777777776, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027777777777777776, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24242424242424243, 0.0, 0.42857142857142855, 0.0, 0.0, 0.20408163265306123]

    acc_fajardo2 = []
    acc_simpler2 = []
    for x, y in zip(cer_fajardo, cer_simpler):
        if x == 0.0:
            acc_fajardo2.append(1)
        else:
            acc_fajardo2.append(0)
        if y == 0.0:
            acc_simpler2.append(1)
        else:
            acc_simpler2.append(0)

    cer_diff = []
    for x, y in zip(acc_fajardo2, acc_simpler2):
        cer_diff.append(x-y)


    time_fajardo = [53.656895, 6094, 102.94]
    time_simpler = [73.182085, 3900, 28.48]

    time_diff = []
    for x, y in zip(time_fajardo, time_simpler):
        time_diff.append(x-y)
    print(time_diff)

    shapiro_statistic, p = shapiro(cer_diff)
    print('Shapiro statistic of the differences in CER:', shapiro_statistic)
    print('p-value of the differences in CER:', p)

    shapiro_statistic, p = shapiro(time_diff)
    print('Shapiro statistic of the differences in time:', shapiro_statistic)
    print('p-value of the differences in CER:', p)

    r = pyasl.generalizedESD(time_diff, 3, 0.05, fullOutput=True)
    print('Number of outliers for the differences in time:',r[0])

    r = wilcoxon(acc_fajardo2,acc_simpler2,  alternative="less")
    print(r)

    r = wilcoxon(time_fajardo, time_simpler, alternative="greater")
    print(r)
