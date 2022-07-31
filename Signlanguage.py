#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from xlrd import open_workbook
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import csv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 


# # Functions for Mediapipe and Extracting Keypoints

# In[4]:


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# # Extract Keypoint Values

# In[60]:


wb = open_workbook(r'F:\Signlanguage\Jupyter\DichNNKH\VideoFileTranslatorList_3_checked_test_2words.xls')
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
column_index = 0
column = sheet.cell_value(0, column_index)

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

for row in range(3, sheet.nrows):
    
    Vid_Link = sheet.cell_value(row, column_index + 3)
    Letter = sheet.cell_value(row, column_index + 5)
    
    File_No = str(sheet.cell_value(row, column_index))
    
#     disallowed_characters = ".?><"
#     for character in disallowed_characters:
#         Letter = Letter.replace(character, "")

    # One video worth of data
    no_sequences = 10

    for sequence in range(no_sequences): #Loop through Videos
            
        npy_path = os.path.join(DATA_PATH, File_No, str(sequence))
        os.makedirs(npy_path)

        cap = cv2.VideoCapture(Vid_Link)

        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:


            Frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(Frame_length)
            print(Letter)

            # Loop through video length aka sequence length
            for frame_num in range(Frame_length):

                # Read feed
                ret, frame = cap.read()
                

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                

                #npy_path = os.path.join(DATA_PATH, File_No, str(sequence), str(frame_num))
                np.save(os.path.join(npy_path, str(frame_num)), keypoints)
                
        cap.release()
        cv2.destroyAllWindows()
        print("Done")


# # Preprocess Data and Create Labels and Features

# In[5]:


Let_List = []
wb = open_workbook(r'F:\Signlanguage\Jupyter\DichNNKH\VideoFileTranslatorList_3_checked_test_2words.xls')
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
column_index = 0
column = sheet.cell_value(0, column_index)
no_sequences = 1

for row in range(3, sheet.nrows):
    Let_Link = sheet.cell_value(row, column_index+5)
    Let_List.append(Let_Link)
Let_Arr = np.array(Let_List)

label_map = {label: num for num, label in enumerate(Let_Arr)}


# In[6]:


#This set of array is required to keep the Numpy array in shape

Null_List = []
Null_Arr = np.array(Null_List)

for n in range(1662): 
    Null_Arr = np.append(Null_Arr, [0])
    n+=1


# In[8]:


sequences, labels = [], []

Null_List = []
Null_Arr = np.array(Null_List)

Null_List = []
Null_Arr = np.array(Null_List)

for n in range(1662): 
    Null_Arr = np.append(Null_Arr, [0])
    n+=1

DATA_PATH = os.path.join('MP_Data')

sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
column_index = 0
column = sheet.cell_value(0, column_index)

for row in range(3, sheet.nrows):
    Vid_Link = sheet.cell_value(row, column_index+3)
    Letter = sheet.cell_value(row, column_index+5)
    File_No = str(sheet.cell_value(row, column_index))

    for sequence in range(no_sequences):
        cap = cv2.VideoCapture(Vid_Link)
        window = []
        Frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_num in range(300):
            try:
                res = np.load(os.path.join(DATA_PATH, File_No, str(sequence), "{}.npy".format(frame_num)))
                
                #print(res)
                #print(len(res))
                window.append(res)
                
            except:
                window.append(Null_Arr)
                
        sequences.append(window)
        labels.append(label_map[Letter])
        #print(sequences,'1')
        #print(labels, '2')
    cap.release()
    cv2.destroyAllWindows()
        
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# # Build and Train LSTM Neural Network

# In[9]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from keras.optimizers import SGD

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(300,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
#model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(Let_Arr.shape[0], activation='softmax'))

res = Let_Arr

opt = SGD(learning_rate = 1)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])


# In[49]:


model.summary()


# In[16]:


res = model.predict(X_test)


# In[17]:


np.sum(res)


# In[14]:


Let_Arr[np.argmax(res[0])]


# In[15]:


Let_Arr[np.argmax(y_test[0])]


# In[10]:


model.save('action.h5') # Save weights


# In[11]:


model.load_weights('action.h5')


# In[14]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[15]:


multilabel_confusion_matrix(ytrue, yhat)


# In[16]:


accuracy_score(ytrue, yhat)


# In[ ]:




