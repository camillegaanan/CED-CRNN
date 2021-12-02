"""Dataset reader and process"""

import os
import html
import h5py
import string
import random
import numpy as np
import multiprocessing
import xml.etree.ElementTree as ET

from pathlib import Path
from glob import glob
from tqdm import tqdm
from data import preproc as pp
from functools import partial
import datetime


class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = getattr(self, f"_{self.name}")()

        if not self.dataset:
            self.dataset = self._init_dataset()

        for y in self.partitions:
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']

    def save_partitions(self, target, image_input_size, max_text_length):
        """Save images and sentences from dataset"""

        os.makedirs(os.path.dirname(target), exist_ok=True)
        total = 0

        with h5py.File(target, "w") as hf:
            for pt in self.partitions:
                self.dataset[pt] = self.check_text(self.dataset[pt], max_text_length)
                size = (len(self.dataset[pt]['dt']),) + image_input_size[:2]
                total += size[0]

                dummy_image = np.zeros(size, dtype=np.uint8)
                dummy_sentence = [("c" * max_text_length).encode()] * size[0]

                hf.create_dataset(f"{pt}/dt", data=dummy_image, compression="gzip", compression_opts=9)
                hf.create_dataset(f"{pt}/gt", data=dummy_sentence, compression="gzip", compression_opts=9)

        pbar = tqdm(total=total)
        batch_size = 1024

        for pt in self.partitions:
            for batch in range(0, len(self.dataset[pt]['gt']), batch_size):
                images = []
                
                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                    start_time = datetime.datetime.now()
                    r = pool.map(partial(pp.preprocess, input_size=image_input_size),
                                 self.dataset[pt]['dt'][batch:batch + batch_size])
                    print(datetime.datetime.now() - start_time)
                    images.append(r)
                    pool.close()
                    pool.join()

                with h5py.File(target, "a") as hf:
                    hf[f"{pt}/dt"][batch:batch + batch_size] = images
                    hf[f"{pt}/gt"][batch:batch + batch_size] = [s.encode() for s in self.dataset[pt]
                                                                ['gt'][batch:batch + batch_size]]
                    pbar.update(batch_size)

    def _init_dataset(self):
        dataset = dict()

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

        return dataset

    def _shuffle(self, *ls):
        random.seed(42)

        if len(ls) == 1:
            li = list(*ls)
            random.shuffle(li)
            return li

        li = list(zip(*ls))
        random.shuffle(li)
        return zip(*li)

    def _doctors(self):
        """Handwritten Prescriptions reader"""

        pt_path =  os.path.join(self.source, "split")

        paths = {"train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "val.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()}

        image_texts = []
        image_name = []
        data_dir = Path(os.path.join(self.source, "images"))
        for filename in os.listdir(data_dir):
            image_name.append(filename)
            if filename.find("Azathioprine") != -1:
                image_texts.append("Azathioprine: 3-5 mg/kg Per os OD")
            elif filename.find("Ceftriaxone") != -1:
                image_texts.append("Ceftriaxone: 2 g IV q24h")
            elif filename.find("Chlorpromazine") != -1:
                image_texts.append("Chlorpromazine: 10-25 mg Per os three times a day")
            elif filename.find("Dobutamine") != -1:
                image_texts.append("Dobutamine: 2.5-15 mcg/kg/min")
            elif filename.find("Hydroxyzine") != -1:
                image_texts.append("Hydroxyzine: 50-100 mg by IJ qds")
            elif filename.find("Lorazepam") != -1:
                image_texts.append("Lorazepam: 1 mg Per os 2 times a day")
            elif filename.find("Metronidazole") != -1:
                image_texts.append("Metronidazole: 7.5 mg/kg Per os q6hr")
            elif filename.find("Prednisolone") != -1:
                image_texts.append("Prednisolone: 5-60 mg per day qds")
            elif filename.find("Quinine") != -1:
                image_texts.append("Quinine: 648 mg Per os every 8 hours for 7 days")
            elif filename.find("Risperidone") != -1:
                image_texts.append("Risperidone: 2 mg orally i/d")
            elif filename.find("Rituximab") != -1:
                image_texts.append("Rituximab: 375 mg/m2 IV once weekly")
            else:
                image_texts.append("Tramadol: 50-100 mg as needed every 4 to 6 hours")

        with open(os.path.join(self.source, "label", "ground_truth.txt"), 'w') as f:
            for name, label in zip(image_name,image_texts):
                f.write(name.replace('.png','') + ' ' + label)
                f.write('\n')

        lines = open(os.path.join(self.source, "label", "ground_truth.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            split = line.split()
            if split[1].find("CHECK") != -1:
                split[0] = split[0] + ' ' + split[1]
                split.pop(1)
            if line.find("Ferraren_Quinine_3") != -1:
                split[0] = split[0] #+ ' ' + split[1]
                #split.pop(1)
            if line.find("Ferraren_Quinine_2") != -1:
                split[0] = split[0] #+ ' ' + split[1]
                #split.pop(1)
            if line.find("Lim_Chlorpromazine_1") != -1:
                split[0] = split[0][0:20] + '  ' + split[0][21:35]
            gt_dict[split[0]] = ''
            for i in range(1, len(split)):
                gt_dict[split[0]] += str(split[i]) + ' '
            gt_dict[split[0]] = gt_dict[split[0]][:len(gt_dict[split[0]])-1]

        img_path = os.path.join(self.source, "images")
        dataset = self._init_dataset()
    
        for i in self.partitions:
            for line in paths[i]:
                if line.find(".txt") == -1:
                    dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                    dataset[i]['gt'].append(gt_dict[line])
        return dataset

    @staticmethod
    def check_text(data, max_text_length=128):
        """Checks if the text has more characters instead of punctuation marks"""

        dt = {'gt': list(data['gt']), 'dt': list(data['dt'])}

        for i in reversed(range(len(dt['gt']))):
            text = pp.text_standardize(dt['gt'][i])
            strip_punc = text.strip(string.punctuation).strip()
            no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

            length_valid = (len(text) > 1) and (len(text) < max_text_length)
            text_valid = (len(strip_punc) > 1) and (len(no_punc) > 1)

            if (not length_valid) or (not text_valid):
                dt['gt'].pop(i)
                dt['dt'].pop(i)
                continue

        return dt
