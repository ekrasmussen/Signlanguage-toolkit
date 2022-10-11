from datetime import datetime
from difflib import restore
from pickletools import optimize
import random
import mediapipe as mp
import cv2 
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, callbacks


mp_holistic = mp.solutions.holistic 

#used for detecting keypoints
def detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = model.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return results

#used for extracting keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

#path for exported data
DATA_PATH = os.path.join('MP_Data') 

#actions we try to detect
actions = np.array(['A', 'B', 'C', 'D', 'E', 'Idle']) 

#number of videos per sign
no_sequences = 50 

#videos should have 40 frames
desired_length = 40

#create folder for each action
for action in actions: 
    #use video in video_list at some point Look down below for help
    for sequence in range (no_sequences):  
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass #THROW EXCEPTIONS HERE!


#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #goes through the actions
    for action in actions:
        #gets list of videos
        video_list = os.listdir(f'Training_videos\{action}')
        #goes through the list of videos
        for video in video_list:
            #grabs video
            cap = cv2.VideoCapture(cv2.VideoCapture(f'Training_videos\{action}\{video}'))
            #counts amount of frames in video
            video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            #if video is shorter than desired amount of frames start frame is set to 0
            if desired_length >= video_length:
                start = 0
            #else starts at a random frame that will results in desired amount of frames
            else:
                max_start = video_length - desired_length
                start = random.randint(0, max_start)
            #sets start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            #goes through frames
            for frame_num in range (desired_length):
                #uses read to get frame
                has_frame, frame = cap.read()
                #if no frame exists, fill in blank
                if not has_frame:
                    frame = cap.imread('black.png')
            #gets results back from detection method
            results = detection(frame, mp_holistic)

            #gets keypoints from extract_keypoints function
            keypoints = extract_keypoints(results)
            #sets path for numpy array
            npy_path = os.path.join(DATA_PATH, action, str(video), str(frame_num))
            #saves numpy array with keypoints
            np.save(npy_path, keypoints)
    #releases the videocapture
    cap.release()

#maps labels to numbers
label_map = {label:num for num, label in enumerate(actions)}

#creates to lists
sequences, labels = [], []
#goes through actions
for action in actions:
    #goes through sequences
    for sequence in range(no_sequences):
        #creates list for of frames for a sequence
        window = []
        #goes through sequences
        for frame_num in range(desired_length):
            #loads numpy array
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            #appends frame to window list
            window.append(res)
        #appends window list to sequences list
        sequences.append(window)
        #appends label map for action to labels list
        labels.append(label_map[action])

#set X for model training
X = np.array(sequences)
#set y for model training
y = to_categorical(labels).astype(int)

#splits dataset in training and test sets
#random_state sets seed value to allow for comparison of different neural networks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#sets path for log file
log_dir = os.path.join('Logs')
#sets condition for earlystopping
#monitor val_loss
#mode is set to min to stop when val_loss stops descreasing 
#patience sets number of epochs after which training will be stopped
#restore_best_weights restores best weights after stopping
earlystopping = callbacks.Earlystopping(monitor = 'val_loss', mode = 'min', patience = 5, restore_best_weights = True)

#sets up sequential layers in neural network
model = Sequential()
#adds a LSTM layer with 64 nodes, returns a sequence, uses relu activation, and input shape
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(desired_length, 1662)))
model.add(LSTM(128, return_sequence=True, activation='relu'))
#note this doesn't return a sequence
model.add(LSTM(64, return_sequences=False, activation='relu'))
#adds dense layer with 64 nodes uses relu activation
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation="relu"))
#adds dense layer with softmax activation to output action
model.add(Dense(actions.shape[0], activation='softmax'))

#compiles model using categorical_crossentropy due to 
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs = 1000, callbacks = [earlystopping]) #Do not specify the batch_size if your data is in the form of a dataset, generators, or keras.utils.Sequence instances

currentDateTime = datetime

model.save(f'{currentDateTime}_action.h5')

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)

confusion_matrix.to_csv(f'{currentDateTime}_confusion_matrix')

accuracy = accuracy_score(ytrue, yhat)

accuracy.to_csv(f'{currentDateTime}_accuracy')