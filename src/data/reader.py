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

    def _hdsr14_car_a(self):
        """ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets dataset reader"""

        dataset = self._init_dataset()
        partition = self._read_orand_partitions(os.path.join(self.source, "ORAND-CAR-2014"), 'a')

        for pt in self.partitions:
            for item in partition[pt]:
                text = " ".join(list(item[1]))
                dataset[pt]['dt'].append(item[0])
                dataset[pt]['gt'].append(text)

        return dataset

    def _hdsr14_car_b(self):
        """ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets dataset reader"""

        dataset = self._init_dataset()
        partition = self._read_orand_partitions(os.path.join(self.source, "ORAND-CAR-2014"), 'b')

        for pt in self.partitions:
            for item in partition[pt]:
                text = " ".join(list(item[1]))
                dataset[pt]['dt'].append(item[0])
                dataset[pt]['gt'].append(text)

        return dataset

    def _read_orand_partitions(self, basedir, type_f):
        """ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets dataset reader"""

        partition = {"train": [], "valid": [], "test": []}
        folder = f"CAR-{type_f.upper()}"

        for i in ['train', 'test']:
            img_path = os.path.join(basedir, folder, f"{type_f.lower()}_{i}_images")
            txt_file = os.path.join(basedir, folder, f"{type_f.lower()}_{i}_gt.txt")

            with open(txt_file) as f:
                lines = [line.replace("\n", "").split("\t") for line in f]
                lines = [[os.path.join(img_path, x[0]), x[1]] for x in lines]

            partition[i] = lines

        sub_partition = int(len(partition['train']) * 0.1)
        partition['valid'] = partition['train'][:sub_partition]
        partition['train'] = partition['train'][sub_partition:]

        return partition

    def _hdsr14_cvl(self):
        """ICFHR 2014 Competition on Handwritten Digit String Recognition in Challenging Datasets dataset reader"""

        dataset = self._init_dataset()
        partition = {"train": [], "valid": [], "test": []}

        glob_filter = os.path.join(self.source, "cvl-strings", "**", "*.png")
        train_list = [x for x in glob(glob_filter, recursive=True)]

        glob_filter = os.path.join(self.source, "cvl-strings-eval", "**", "*.png")
        test_list = [x for x in glob(glob_filter, recursive=True)]

        sub_partition = int(len(train_list) * 0.1)
        partition['valid'].extend(train_list[:sub_partition])
        partition['train'].extend(train_list[sub_partition:])
        partition['test'].extend(test_list[:])

        for pt in self.partitions:
            for item in partition[pt]:
                text = " ".join(list(os.path.basename(item).split("-")[0]))
                dataset[pt]['dt'].append(item)
                dataset[pt]['gt'].append(text)

        return dataset

    def _bentham(self):
        """Bentham dataset reader"""

        source = os.path.join(self.source, "BenthamDatasetR0-GT")
        pt_path = os.path.join(source, "Partitions")

        paths = {"train": open(os.path.join(pt_path, "TrainLines.lst")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "ValidationLines.lst")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "TestLines.lst")).read().splitlines()}

        transcriptions = os.path.join(source, "Transcriptions")
        gt = os.listdir(transcriptions)
        gt_dict = dict()

        for index, x in enumerate(gt):
            text = " ".join(open(os.path.join(transcriptions, x)).read().splitlines())
            text = html.unescape(text).replace("<gap/>", "")
            gt_dict[os.path.splitext(x)[0]] = " ".join(text.split())

        img_path = os.path.join(source, "Images", "Lines")
        dataset = self._init_dataset()

        for i in self.partitions:
            for line in paths[i]:
                dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                dataset[i]['gt'].append(gt_dict[line])

        return dataset

    def _iam(self):
        """IAM dataset reader"""

        pt_path = os.path.join(self.source, "largeWriterIndependentTextLineRecognitionTask")
        paths = {"train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "validationset1.txt")).read().splitlines() +
                 open(os.path.join(pt_path, "validationset2.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ascii", "lines.txt")).read().splitlines()
        dataset = self._init_dataset()
        gt_dict = dict()

        for line in lines:
            if (not line or line[0] == "#"):
                continue

            split = line.split()
            gt_dict[split[0]] = " ".join(split[8::]).replace("|", " ")

        for i in self.partitions:
            for line in paths[i]:
                try:
                    split = line.split("-")
                    folder = f"{split[0]}-{split[1]}"

                    img_file = f"{split[0]}-{split[1]}-{split[2]}.png"
                    img_path = os.path.join(self.source, "lines", split[0], folder, img_file)

                    dataset[i]['gt'].append(gt_dict[line])
                    dataset[i]['dt'].append(img_path)
                except KeyError:
                    pass

        return dataset

    def _washington(self):
        """Washington dataset reader"""

        pt_path = os.path.join(self.source, "sets", "cv1")

        paths = {"train": open(os.path.join(pt_path, "train.txt")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "valid.txt")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.txt")).read().splitlines()}

        lines = open(os.path.join(self.source, "ground_truth", "transcription.txt")).read().splitlines()
        gt_dict = dict()

        for line in lines:
            split = line.split()
            split[1] = split[1].replace("-", "").replace("|", " ")
            split[1] = split[1].replace("s_pt", ".").replace("s_cm", ",")
            split[1] = split[1].replace("s_mi", "-").replace("s_qo", ":")
            split[1] = split[1].replace("s_sq", ";").replace("s_et", "V")
            split[1] = split[1].replace("s_bl", "(").replace("s_br", ")")
            split[1] = split[1].replace("s_qt", "'").replace("s_GW", "G.W.")
            split[1] = split[1].replace("s_", "")
            gt_dict[split[0]] = split[1]

        img_path = os.path.join(self.source, "data", "line_images_normalized")
        dataset = self._init_dataset()

        for i in self.partitions:
            for line in paths[i]:
                dataset[i]['dt'].append(os.path.join(img_path, f"{line}.png"))
                dataset[i]['gt'].append(gt_dict[line])

        return dataset

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
            if line.find("PayuranGatchalian_Lorazepam_3") != -1:
                split[0] = split[0] + ' ' + split[1]
                split.pop(1)
                split.pop(2)
            if line.find("Ferraren_Quinine_3") != -1:
                split[0] = split[0] + ' ' + split[1]
                split.pop(1)
            if line.find("Ferraren_Quinine_2") != -1:
                split[0] = split[0] + ' ' + split[1]
                split.pop(1)
            if line.find("Lim_Chlorpromazine_1") != -1:
                print(line)
                split[0] = split[0][0:20] + '  ' + split[0][21:35]
                print(split)
            gt_dict[split[0]] = ''
            for i in range(1, len(split)):
                gt_dict[split[0]] += str(split[i]) + ' '
            gt_dict[split[0]] = gt_dict[split[0]][:len(gt_dict[split[0]])-1]

        img_path = os.path.join(self.source, "images")
        dataset = self._init_dataset()
    
        for i in self.partitions:
            for line in paths[i]:
                # print(line)
                if line.find(".txt") == -1:
                    # print(gt_dict[line])
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
