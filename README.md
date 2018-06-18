# Biometric Insurance

This program takes in a selfie image, predicts the age and gender of the person in the image and returns the premium based on the predicted age and gender.

## Structure
    .
	├── ...
    ├── models        
    │   	├── asian_weights.hdf5					# Training result model on Asian faces dataset
    │   	├── shape_predictor_68_face_landmarks.dat 		# Model to crop and align faces in image     
    │   	└── weights.18-4.06.hdf5				# Pre-trained weights
    └── age_predictor.py						# Returns predicted age
    └── gender_predictor.py						# Returns name and predicted gender (name is the image file name 														without extensions)
    └── gui.py							# Starts the GUI for convenient use
    └── wide_resnet.py						# Returns a Wide Residual Network
    └── fine_tune_resnet.py						# Fine-tune model with Asian faces dataset
	
## Prerequisites

Before running any of these files, make sure that you have the required libraries and dependents:

### Python

Go to https://www.python.org/ and download the latest Python version (at least 3.6)

### TensorFlow, Keras and relevant libraries

From CMD or PowerShell, type these following commands one by one:
* TensorFlow
```
pip install tensorflow
```
* Keras
```
pip install Keras
```
* dlib
```
pip install dlib
```
* CV2
```
pip install cv2
```
* imutils
```
pip install imutils
```
* numpy
```
pip install numpy
```
* wxpython
```
pip install wxpython
```
* requests
```
pip install requests
```
* flask
```
pip install flask
```

If the program requires any additional libraries, please install them accordingly using 'pip install' as above.

## Run demo

### Biometric Insurance demo

From **Command Prompt** or **PowerShell** type the following lines:

```
cd ./BiometricInsurance
python gui.py
```

Choose **Browse** to select a selfie image. This may take as long as 10 seconds for the program to predict the age and gender and send to the API to retrieve the premium of the target. Make sure that the API is on by the time you demonstrate.

The **Name** field value is the selfie image file name after removing the extensions and all spacing symbols.

Example result:

![](/images/demo.PNG?raw=true)

### Fine-tune Wide Residual Network demo

For the dataset, let's assume that you have 70 age groups and you have organized your folders into the following order:

	.
	├── ...
    └── dataset        
		└── training
			└────── 1
				├── PIC_01.jpg
				├── PIC_02.jpg
				├── PIC_03.jpg
				├── ...
			└────── 2
				├── PIC_01.jpg
				├── PIC_02.jpg
				├── PIC_03.jpg
				├── ...
			└────── ...
			
			└────── 70
				├── PIC_01.jpg
				├── PIC_02.jpg
				├── PIC_03.jpg
				├── ...
		└── testing
			└────── 1
				├── PIC_01.jpg
				├── PIC_02.jpg
				├── PIC_03.jpg
				├── ...
			└────── 2
				├── PIC_01.jpg
				├── PIC_02.jpg
				├── PIC_03.jpg
				├── ...
			└────── ...
	
			└────── 70
				├── PIC_01.jpg
				├── PIC_02.jpg
				├── PIC_03.jpg
				├── ...

Let's start retraining with the following code:

```
cd ./BiometricInsurance
python fine_tune_resnet.py --train ./dataset/training --valid ./dataset/testing --model ./models/weights.hdf5 --classes 70
```

