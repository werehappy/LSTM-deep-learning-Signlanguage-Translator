# LSTM-deep-learning-Signlanguage-Translator
The purpose of this program is to translate Vietnamese sign language on real time.

This program use Mediapipe model and Tensorflow deep learning model from Google.

It reads the video file with Mediapipe Holistic model to extract keypoints. The keypoints are read by every frames and stored using Numpy arrays.

Creates set of Numpy arrays for the Tensorflow keras model training

Train the model


#Problems
The accuracy of the model does not increase. It seems it does not train well. Shape of the Numpy array for the model training is also a problem
Both problem results the program unusable
