
import os
from data.generator import Tokenizer
import data.preproc as pp
import numpy as np
from network.model import HTRModel
import matplotlib.pyplot as plt
import cv2
from data import evaluation
import string

arch = "fajardo"
source = "doctors"

source_path = os.path.join("..", "data", f"{source}.hdf5")
output_path = os.path.join("..", "output", source, arch)
target_path = os.path.join(output_path, "checkpoint_weights.hdf5")

charset_base = string.printable[:95]
max_text_length = 128
input_size = (1024, 128, 1)
# uploaded = files.upload()
filename = 'C:/Users/Camille/source/repos/CED-CRNN/raw/doctors/images/Yopo_Tramadol_1.png'
filename = 'C:/Users/Camille/Downloads/Metronidazole.jpg'

tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)

img = pp.preprocess(filename, input_size=input_size)
x_test = pp.normalization([img])

model = HTRModel(architecture=arch,
                 input_size=input_size,
                 vocab_size=tokenizer.vocab_size,
                 beam_width=10,
                 top_paths=10)

model.compile(learning_rate=0.001)
model.load_checkpoint(target=target_path)

predict, _ = model.predict(x_test, ctc_decode=True)
predict = [[tokenizer.decode(x) for x in y] for y in predict]

if filename.find("Azathioprine") != -1:
  ground_truth = "Azathioprine: 3-5 mg/kg Per os OD"
elif filename.find("Ceftriaxone") != -1:
  ground_truth = "Ceftriaxone: 2 g IV q24h"
elif filename.find("Chlorpromazine") != -1:
  ground_truth = "Chlorpromazine: 10-25 mg Per os three times a day"
elif filename.find("Dobutamine") != -1:
  ground_truth = "Dobutamine: 2.5-15 mcg/kg/min"
elif filename.find("Hydroxyzine") != -1:
  ground_truth = "Hydroxyzine: 50-100 mg by IJ qds"
elif filename.find("Lorazepam") != -1:
  ground_truth = "Lorazepam: 1 mg Per os 2 times a day"
elif filename.find("Metronidazole") != -1:
  ground_truth = "Metronidazole: 7.5 mg/kg Per os q6hr"
elif filename.find("Prednisolone") != -1:
  ground_truth = "Prednisolone: 5-60 mg per day qds"
elif filename.find("Quinine") != -1:
  ground_truth = "Quinine: 648 mg Per os every 8 hours for 7 days"
elif filename.find("Risperidone") != -1:
  ground_truth = "Risperidone: 2 mg orally i/d"
elif filename.find("Rituximab") != -1:
  ground_truth = "Rituximab: 375 mg/m2 IV once weekly"
else:
  ground_truth = "Tramadol: 50-100 mg as needed every 4 to 6 hours"

print("Preprocessed Image:")
cv2.imshow('Preprocessed image', pp.adjust_to_see(img))
cv2.waitKey()
print("Expected:", ground_truth)
print("Predicted:", predict[0][0])
if ground_truth == predict[0][0]:
  print("Correctly recognized")
else:
  print("Incorrectly recognized")

print("\nOriginal Image:")
raw_image = cv2.imread(filename,cv2.COLOR_BGR2RGB)
cv2.imshow('Raw image', raw_image)
cv2.waitKey()

_, metrics = evaluation.ocr_metrics([predict[0][0]], [ground_truth], output_path)

print("Character Error Rate:", metrics[0])
print("Word Error Rate:", metrics[1])
print("Word Accuracy:", metrics[2])
print("Word Recall:", metrics[3])
print("Word Precision:", metrics[4])
print("Word F1 Score:", metrics[5])