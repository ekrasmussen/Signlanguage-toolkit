import cv2
from load_model import *
from detect import *
import mediapipe as mp
from datetime import datetime
import os

class VideoReader:

    #Constructor
    def __init__(self, file_path, video_path):
            self.cap = cv2.VideoCapture(video_path)
            self.mp_holistic = mp.solutions.holistic
            self.mp_drawing = mp.solutions.drawing_utils
            self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.model = Model(file_path)
            self.actions = self.model.actions

    #Gets image and predicts sign
    #Requirement is for the corresponding h5 file has a ini file of same name
    def start(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            
            #Checks if there is a image, if not break the loop
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.holistic.process(image)

                keypoints = extract_hand_keypoints(results)

                is_desired_length, res = self.model.predict(keypoints)

                if is_desired_length:
                    self.model.sentence_update(res)

            else:
                break

        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
        self.save_to_text(dt_string)
        return dt_string #returns so tests can get the txt file

    #Saves the sentence in a txt document
    def save_to_text(self, dt_string):
        with open(os.path.join('Sentences', f"Sentence_{dt_string}.txt"), 'a') as f:
            f.write(f'{self.model.sentence} \n')
            f.close()