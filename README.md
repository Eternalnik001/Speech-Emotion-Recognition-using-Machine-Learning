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

7.7 Compile the Model
To compile the model we need to define loss function which is categorical cross-entropy, accuracy 
metrics which is accuracy score, and an optimizer which is Adam.


7.8 Train the Model
We will train the model and save the model in HDF5 format. We will train a model for 100 epochs 
and batch size as 32. We’ll use callback, which is a checkpoint to know how much time it took to 
trail over data’

7.9 Testing with Actual and Predicted Test Lables

7.10 Check the Test Accuracy
Finally, we will use the evaluate() method to assess the trained model on the test set and determine 
the model's correctness or Accuracy.




