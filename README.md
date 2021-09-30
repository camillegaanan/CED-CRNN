Process:
1. Pull changes from main branch.<br>
	-NOTE: check src > network > model.py > def cnn_bilstm<br>
	-if gusto mo nung og model na 5min param, comment out na lng yung nasa baba na for the simpler model.<br>
	-vice versa if gusto mo uis yung simpler model<br>
2. add an empty folder called "data"
3. Add an empty folder called "raw"
4. Go to src>data>DataAugmentation.ipynb> Run cell#1.<br>
	-input_path = path containing the pictures to be augmented<br>
	-save_to_dir = path where the augmented images are going to be saved
5. Split images using DataAugmentation.ipynb cell#2.<br>
	-root_dir = path kung san ilalagay yung splitted images<br>
	-src = path of augmented images
6. Add "doctors" folder in raw
7. Add 3 subfolder: images, label, split<br>
	-images: all of the images<br>
	-split: txt files containing the file names of the images for test,train, and validation<br>
		- how to generate txt file:<br>
			a.Open command prompt and change directory to the folder containing the training images<br>
			b.run: dir/b>train.txt<br>
			c. Repeat steps a-b with test and validation images (change filename into test.txt and val.txt)<br>
			d. Open EACH txt file using Notepad++. Click Ctrl+F, go to the Replace tab, Find .jpg and replace with nothing. Click replace all.<br>
			e. mapupunta sa train, test, and split folders yung txt file. cut mo sila palabas and transfer to split folder na ginawa kanina.<br>
	-label: create an empty txt file with file name "ground_truth.txt"
8. Go to cmd and navigate to location of project folder.<br>
	- Run: python -m venv .venv && .venv\Scripts\activate<br>
	- Run: pip install -r requirements.txt<br>
	- Run: python main.py --source=doctors --transform<br>
	- If no module found error occured, install the packages while the virtual environment is still activated.<br>
	- After running the codes, Run: deactivate and exit the cmd.<br>
9. Upload "data" and "src" folder in the SAME folder of your Google Drive.
10. Run 2 Google Drive Environment and below on Google Colab<br>
<br>
Changes for tutorial.ipynb:<br>
- 2.2 Google Drive > change %cd ".gdrive/MyDrive/xxxx to the location of your src folder<br>
- 3.1 Environment<br>
	- source="doctors"<br>
	- arch="cnn_bilstm"<br>
	- magdagdag ng folder name sa dulo ng output_path. example:<br>
		output_path = os.path.join("..", "output", source, arch, "without preprocessing")<br>
- 3.3 HTRModel Class<br>
	- add the following inside model.compile, after learnig_rate<br>
		initial_step=0, target="model.jpg", output=output_path<br>
- 4 Training<br>
	- Go to cell 1 and click " + Code" (nasa upper left)<br>
	- paste this to the newly added cell<br>
	<br>
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
