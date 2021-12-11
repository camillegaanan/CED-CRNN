A thesis by Ashnee Gaile C. Cabais, Camille Dawn C. Gaanan, Trixcie Jane S. Mirasol, and Carl Joseph A. Paez.

Handwritten Text Recognition (HTR) system implemented using Tensorflow and trained on gathered doctors' cursive handwriting images. This Neural Network model recognizes the text contained in the images of segmented texts lines, and uses Compass Edge Detection as a preprocessing method.

## Preparation of System:
On localhost:
- Download the zip file `dataset.zip` and `main.zip` and extract all in your preferred location. 
- Download the zip file `main.zip` and extract all in your preferred location. Your project directory for the will be like this:
.
├── data
│
├── raw
│   ├── doctors
│   │   ├── images
│   │   ├── label
│   │   ├── split
└── src
    ├── data
    │   ├──DataAugmentation.ipynb
    │   ├── evaluation.py
    │   ├── generator.py
    │   ├── preproc.py
    │   ├── reader.py
    ├── main.py
    ├── network
    │   ├── model.py
    └── tutorial.ipynb

In DataAugmentation.ipynb, change the strings of `src`, `parent`, and `aug_dir` such that:
    - src = location of the extracted files from `dataset.zip`.
    - parent = location of the raw > doctors > images subfolder. Take note that the string has an addition `/` at the end.
    - train_aug = location of an empty folder that will contain the augmented training images later on.

Run the first and only cell in DataAugmentation.ipynb. This will create two new csv files named `data.csv` and `train_aug.csv` under the  same hierarchy of the notebook. Three text files named `train.txt`,`val.txt`, and `test.txt` will also be created under raw > doctors > split.

On your command line, navigate to the location where you extracted `main.zip`. 

After that, create virtual environment and install the dependencies with python 3 and pip:
`python -m venv .venv && .venv\Scripts\activate`
`pip install -r requirements.txt`

Go to `src` and run `python main.py --source=doctors --transform` to create the hdf5 file containing the images and their corresponding groud truth.

If no module found error occurred, install the packages while the virtual environment is still activated using the `pip install` command.

After creating the hdf5 file, this wll save under the data subfolder. Run `Deactivate` and close the project.

In your Google Drive, create a folder containing the `data` and `src` subfolders of the project.

Open `main.ipynb` on your Google Colab, ensuring that it is using GPU. At 1.2 Google Drive, make sure to modify the grive location to your directory.  

## Training the System and Testing it by Batch
Run `main.ipynb` on your Google Colab to train and test the dataset. Feel free to modify the epochs, batch size, input size, beam width, stop tolerance, and redce tolerance to your liking.