import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

mp_holistic = mp.solutions.holistic  # Det er her vi introducerer vores holistiske model
mp_drawing = mp.solutions.drawing_utils  # Her kommer der en mulighed for at tegne ved hjælp af drawing utilities


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


def extract_keypoints_hands(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([lh, rh])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image,
                         cv2.COLOR_BGR2RGB)  # Her skifter vi farven på billedet fra Blå,grøn og rød til Rød, grøn og blå
    image.flags.writeable = False  # Vi gør image unwriteable
    results = model.process(image)
    image.flags.writeable = True  # Vi gør image writeable igen
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Her skifter vi det tilbage igen
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_TESSELATION)  # Tegner ansigtet med landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)  # Tegner din posering med landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Tegner venstre hånd med landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Tegner højre hånd med landmarks


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              # Tegner din posering med landmarks
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


actions = np.array(['A', 'B', 'C', 'D', 'E', 'Idle'])


def load_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.summary()
    model.load_weights('epoch_400_frames_10.h5')
    return model


colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (45, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame

def prob_text(res):
    predictions = []
    for num, prob in enumerate(res):
        predictions.append(f'{int(prob * 100)}')

    return predictions

def start_webcam():
    cap = cv2.VideoCapture(0)



def start_read(keypoints, image, sequence, sentence, model):
    # 1. New detection variables
    threshold = 0.8
    predictions = []

   
    sequence.append(keypoints)
    sequence = sequence[-10:]
    if len(sequence) == 10:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # if len(sequence) == 10:
                # res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(actions[np.argmax(res)])

                # 3. Viz logic
        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

                # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            predictions = prob_text(res)


    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image, sentence, predictions
