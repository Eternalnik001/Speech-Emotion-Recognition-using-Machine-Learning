7.1 Dataset
we developed the system to get the input dataset for the training and testing purpose. Dataset is 
given in the model folder. The dataset consists of 2,800 Speech Emotion audio dataset. The dataset 
consist of classes like: angry, disgust, Fear, happy, neutral, Sad and surprise. The dataset is referred 
from the kaggle website. 
Kaggle Link: https://www.kaggle.com/datasets/jayaprakashpondy/speech-emotion-dataset


7.2 Importing the necessary libraries
we import the necessary libraries for our speech emotion detection system. The very important and 
library that supports audio and music analysis is Librosa. It provides building blocks that are 
required to construct an information retrieval model from music. Another library we will use is for 
deep learning modeling purposes is TensorFlow
import IPython.display as ipd
import librosa
import librosa.display
import pandas as pd
import os, time, warnings
import seaborn as sns
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
 Dense,
 Conv1D,
 MaxPooling1D,
 BatchNormalization,
 Dropout, 
 Flatten,
 Conv2D,
 MaxPool2D,
)
from IPython.display import Audio
from sklearn.metrics import confusion_matrix, classification_report


7.3 Exploratory Data Analysis of Audio data
We have different folders under the dataset folder. Before applying any preprocessing, we will try 
to understand how to load audio files and how to visualize them in form of the waveform. If you 
want to load the audio file and listen to it, then you can use the IPython library and directly give it 
an audio file path. We have taken the first audio file in the fold 1 folder.
Now we will use Librosa to load audio data. So when we load any audio file with Librosa, it gives 
us 2 things. One is sample rate, and the other is a two-dimensional array. Let us load the above 
audio file with Librosa and plot the waveform using Librosa.
Sample rate – It represents how many samples are recorded per second. The default sampling rate 
with which librosa reads the file is 2,800.The sample rate differs by the library you choose.
2-D Array – The first axis represents recorded samples of amplitude. And the second axis 
represents the number of channels. There are different types of channels – Monophonic(audio that 
has one channel) and stereo(audio that has two channels).
We load the data with librosa, then it normalizes the entire data and tries to give it in a single 
sample rate. 
Librosa is popular for
It tries to converge the signal into mono(one channel).
• It can represent the audio signal between -1 to +1(in normalized form), so a regular pattern 
is observed.
• It is also able to see the sample rate, and by default, it converts it to 22 kHz, while in the 
case of other libraries, we see it according to a different value

7.4 Imbalance Dataset check
we know about the audio files and how to visualize them in audio format. Moving format to data 
exploration we will load the CSV data file provided for each audio file and check how many 
records we have for each class.
The data we have is a filename and where it is present so let us explore 1st file, so it is present in 
fold 7 with category. By use of value counts function to check records of each class.
We can see that output data is not imbalanced, and most of the classes have an approximately equal 
number of records

7.5 Data Preprocessing
Some audios are getting recorded at a different rate-like 44KHz or 22KHz. Using librosa, it will 
be at 22KHz, and then, we can see the data in a normalized pattern. Now, our task is to extract 
some important information, and keep our data in the form of independent(Extracted features from 
the audio signal) and dependent features(class labels). We will use Mel Frequency Cepstral 
coefficients to extract independent features from audio signals.
MFCCs – The MFCC summarizes the frequency distribution across the window size. So, it is 
possible to analyze both the frequency and time characteristics of the sound. This audio 
representation will allow us to identify features for classification. So, it will try to convert audio 
into some kind of features based on time and frequency characteristics that will help us to do 
classification Now, we have to extract features from all the audio files and prepare the data frame. 
So, we will create a function that takes the filename (file path where it is present). It loads the file 
using librosa, where we get 2 information. First, we’ll find MFCC for the audio data, and to find 
out scaled features, we’ll find the mean of the transpose of an array. Now, to extract all the features 
for each audio file, we have to use a loop over each row in the data frame. We also use the TQDM 
python library to track the progress. Inside the loop, we’ll prepare a customized file path for each 
file and call the function to extract MFCC features and append features and corresponding labels 
in a newly formed data frame.
def features_extractor(file):
 audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
 mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
 mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
 return mfccs_scaled_features
extracted_features=[]
for index_num,row in tqdm(Tess_df.iterrows()):
 file_name = os.path.join(os.path.abspath(base_dir),str(row["Path"]))
 final_class_labels=row["Emotions"]
 data=features_extractor(file_name)
 extracted_features.append([data,final_class_labels])
 
7.6 Ann Model Creation
Split the dataset into train and test. 80% train data and 20% test data.
We have extracted features from the audio sample and splitter in the train and test set. Now we 
will implement an ANN model using Keras sequential API. The number of classes is 7, which is 
our output shape (number of classes), and we will create ANN with 3 dense layers and architecture 
is explained below.
• The first layer has 100 neurons. Input shape is 40 according to the number of features with 
activation function as Relu, and to avoid any overfitting, we’ll use the Dropout layer at a 
rate of 0.5.
• The second layer has 200 neurons with activation function as Relu and the drop out at a 
rate of 0.5.
• The third layer again has 100 neurons with activation as Relu and the drop out at a rate of 
0.5.
X_train, X_test, y_train, y_test = train_test_split(
 X, Y, test_size=0.1, random_state=42
)
# print the details
print("Number of training samples = ", X_train.shape[0])
print ("Number of testing samples = ", X_test.shape[0])
num_labels = Y.shape[1]
ANN_Model = Sequential()
ANN_Model.add(Dense(1000, activation="relu", input_shape=(40,)))
ANN_Model.add(Dense(750, activation="relu"))
ANN_Model.add(Dense(500, activation="relu"))
ANN_Model.add(Dense(250, activation="relu"))
ANN_Model.add(Dense(100, activation="relu"))
ANN_Model.add(Dense(50, activation="relu"))
ANN_Model.add(Dense(num_labels, activation="softmax"))
ANN_Model.summary()

7.7 Compile the Model
To compile the model we need to define loss function which is categorical cross-entropy, accuracy 
metrics which is accuracy score, and an optimizer which is Adam.
ANN_Model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

7.8 Train the Model
We will train the model and save the model in HDF5 format. We will train a model for 100 epochs 
and batch size as 32. We’ll use callback, which is a checkpoint to know how much time it took to 
trail over data’
num_epochs = 100
num_batch_size = 32
t0 = time.time()
history = ANN_Model.fit(
 X_train,
 y_train,
 batch_size=num_batch_size,
 epochs=num_epochs,
 validation_data=(X_test, y_test),
)
ANN_Model.save("Model1.h5")
print("ANN Model Saved")

7.9 Check the Test Accuracy
Finally, we will use the evaluate() method to assess the trained model on the test set and determine 
the model's correctness or Accuracy.




