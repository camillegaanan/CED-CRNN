Process:
1. Pull changes from main branch.
	-NOTE: check src > network > model.py > def cnn_bilstm
	-if gusto mo nung og model na 5min param, comment out na lng yung nasa baba na for the simpler model.
	-vice versa if gusto mo uis yung simpler model
2. add an empty folder called "data"
3. Add an empty folder called "raw"
4. Go to src>data>DataAugmentation.ipynb> Run cell#1.
	-input_path = path containing the pictures to be augmented
	-save_to_dir = path where the augmented images are going to be saved
5. Split images using DataAugmentation.ipynb cell#2.
	-root_dir = path kung san ilalagay yung splitted images
	-src = path of augmented images
6. Add "doctors" folder in raw
7. Add 3 subfolder: images, label, split
	-images: all of the images
	-split: txt files containing the file names of the images for test,train, and validation
		- how to generate txt file:
			a.Open command prompt and change directory to the folder containing the training images
			b.run: dir/b>train.txt
			c. Repeat steps a-b with test and validation images (change filename into test.txt and val.txt)
			d. Open EACH txt file using Notepad++. Click Ctrl+F, go to the Replace tab, Find .jpg and replace with nothing. Click replace all.
			e. mapupunta sa train, test, and split folders yung txt file. cut mo sila palabas and transfer to split folder na ginawa kanina.
	-label: create an empty txt file with file name "ground_truth.txt"
8. Go to cmd and navigate to location of project folder.
	- Run: python -m venv .venv && .venv\Scripts\activate
	- Run: pip install -r requirements.txt
	- Run: python main.py --source=doctors --transform
	- If no module found error occured, install the packages while the virtual environment is still activated.
	- After running the codes, Run: deactivate and exit the cmd.
9. Upload "data" and "src" folder in the SAME folder of your Google Drive.
10. Run 2 Google Drive Environment and below on Google Colab

Changes for tutorial.ipynb:
- 2.2 Google Drive > change %cd ".gdrive/MyDrive/xxxx to the location of your src folder
- 3.1 Environment
	- source="doctors"
	- arch="cnn_bilstm"
	- magdagdag ng folder name sa dulo ng output_path. example:
		output_path = os.path.join("..", "output", source, arch, "without preprocessing")
- 3.3 HTRModel Class
	- add the following inside model.compile, after learnig_rate
		initial_step=0, target="model.jpg", output=output_path
- 4 Training
	- Go to cell 1 and click " + Code" (nasa upper left)
	- paste this to the newly added cell
	
from matplotlib import pyplot as plt

loss = h.history['loss']
val_loss = h.history['val_loss']

epochs_range = range(120)

plt.figure(figsize=(30, 10))

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

NOTE: change epochs_range = range(CHANGE HERE)
CHANGE HERE = number of total epochs na nagawa after training
