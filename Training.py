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

def detection(frame, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


DATA_PATH = os.path.join('MP_Data') #Path for exported data

actions = np.array(['A', 'B', 'C', 'D', 'E', 'Idle']) #Actions we try to detect

no_sequences = 50 # Number of videos per sign

desired_length = 40 # Videos should have 40 frames

start_folder = 50


for action in actions: #create folder for each action
    for sequence in range (no_sequences): #use video in video_list at some point Look down below for help 
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass # Through at some point plssssssssssssssssss



with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    for action in actions:
        
        video_list = os.listdir(f'Training_videos\{action}')

        for video in video_list:

            cap = cv2.VideoCapture(cv2.VideoCapture(f'Training_videos\{action}\{video}'))

            video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if desired_length >= video_length:
                start = 0
            else:
                max_start = video_length - desired_length
                start = random.randint(0, max_start)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            for frame_num in range (desired_length):

                has_frame, frame = cap.read()
                
                if not has_frame:
                    frame = cap.imread('black.png')
                
            results = detection(frame, mp_holistic)

            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(video), str(frame_num))
            np.save(npy_path, keypoints)

    cap.release()

label_map = {label:num for num, label in enumerate(actions)}

print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(desired_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_dir = os.path.join('Logs')
earlystopping = callbacks.Earlystopping(monitor = 'val_loss', mode = 'min', patience = 5, restore_best_weights = True)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(desired_length, 1662)))
model.add(LSTM(128, return_sequence=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation='softmax'))

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