import imp
import mediapipe as mp
import cv2 
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

mp_holistic = mp.solutions.holistic

def detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

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
    for sequnce in range (no_sequences): #use video in video_list at some point Look down below for help 
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequnce)))
        except:
            pass # Through at some point plssssssssssssssssss



with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    for action in actions:
        
        video_list = os.listdir(f'Training_videos\{action}')


        for video in video_list:

            cap = cv2.VideoCapture(f'Training_videos\{action}\{video}')

            for frame_num in range (desired_length):

                has_frame, frame = cap.read()

                if not has_frame:
                    print('uup or oop')
                    #make it get black frame

