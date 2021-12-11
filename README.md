A thesis by Ashnee Gaile C. Cabais, Camille Dawn C. Gaanan, Trixcie Jane S. Mirasol, and Carl Joseph A. Paez.

Handwritten Text Recognition (HTR) system implemented using Tensorflow and trained on gathered doctors' cursive handwriting images. This Neural Network model recognizes the text contained in the images of segmented texts lines, and uses Compass Edge Detection as a preprocessing method.

## Preparation of System:
On localhost:
- Download the zip file `dataset.zip`, `CED-CRNN.zip`, and `flask-1.zip` and extract all in your preferred location. 
- Your project directory for the will be like this for the extracted files from `CED-CRNN.zip`:
  
.  
├── .venv   
├── data   
├── raw  
│   ├── doctors  
│   │   ├── images  
│   │   ├── label  
│   │   └── split  
└── src  
    ├── data  
    │   ├──DataAugmentation.ipynb  
    │   ├── evaluation.py  
    │   ├── generator.py  
    │   ├── preproc.py   
    │   └── reader.py   
    ├── network  
    │   └── model.py  
    ├── main.py
    └── main.ipynb  

In DataAugmentation.ipynb, change the strings of `src`, `parent`, and `aug_dir` such that:  
    - src = location of the extracted files from `dataset.zip`.  
    - parent = location of the raw > doctors > images subfolder. Take note that the string has an addition `/` at the end.  
    - train_aug = location of an empty folder that will contain the augmented training images later on.  

Run the first and only cell in DataAugmentation.ipynb. This will create two new csv files named `data.csv` and `train_aug.csv` under the  same hierarchy of the notebook. Three text files named `train.txt`,`val.txt`, and `test.txt` will also be created under raw > doctors > split.

On your command line, navigate to the location where you extracted `CED-CRNN.zip`. 

After that, open the virtual environment with:  
`.venv\Scripts\activate`    

Go to `src` and run `python main.py --source=doctors --transform` to create the hdf5 file containing the images and their corresponding groud truth.

If no module found error occurred, install the packages while the virtual environment is still activated using the `pip install` command.

After creating the hdf5 file, this will be saved under the data subfolder. Run `Deactivate` and close the project.

In your Google Drive, create a empty folder where you will upload the `data` and `src` subfolders of the project.

Open `main.ipynb` on your Google Colab, ensuring that it is using GPU. At 1.2 Google Drive, make sure to modify the Google Drive location to your directory.  

## Training the System and Testing it by Batch
Run `main.ipynb` on your Google Colab to train and test the dataset. Feel free to modify the epochs, batch size, input size, beam width, stop tolerance, and redce tolerance to your liking.

## Using the GUI
On localhost, your project directory will be like this for the extracted `flask-1.zip`:

.  
├── env   
├── static   
├── templates    
│   └── home.html    
├── cabais_noNoiseSkeleton.hdf5   
├── crnn_ced   
├── generator.py   
├── loadModel.py   
├── model.py   
└── preproc.py 

In loadModel.py, change the target_path in line 9 such that it will be the location of cabais_noNoiseSkeleton.hdf5 in your directory. 

In crnn_ced.py, change the locations in line 21 to match the location of the raw > images > doctors subfolder. Take note that the string has an addition `/` at the end. Also change line 63 to match the location of the static subfolder.

On your command line, navigate to the location where you extracted `flask-1.zip`. 

After that, open the virtual environment with:  
`env\Scripts\activate`  

And run the command `python crnn_ced.py`. If no module found error occurred, install the packages while the virtual environment is still activated using the `pip install` command.

This will generate a link in your command line. Follow that link to run the GUI of the system.

## Acknowledgement
This software is based on the public repository of [Arthur Flôr](https://github.com/arthurflor23/handwritten-text-recognition).
