import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic  # Det er her vi introducerer vores holistiske model
mp_drawing = mp.solutions.drawing_utils  # Her kommer der en mulighed for at tegne ved hjælp af drawing utilities

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # Her skifter vi farven på billedet fra Blå,grøn og rød til Rød, grøn og blå
    image.flags.writeable = False  # Vi gør image unwriteable
    results = model.process(image)
    image.flags.writeable = True  # Vi gør image writeable igen
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Her skifter vi det tilbage igen
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Tegner ansigtet med landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Tegner din posering med landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Tegner venstre hånd med landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Tegner højre hånd med landmarks

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(80,110, 10), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1,circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,  # Tegner din posering med landmarks
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

cap = cv2.VideoCapture(0)
# Her tilføjer vi mediapipe modellen
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()  # Læser feeden
        image, results = mediapipe_detection(frame, holistic)
        #print(len(results.left_hand_landmarks.landmark))
        print(results)

        draw_styled_landmarks(image, results) # Her tegner den landmarks på webcammet og de bliver rendered
        #Viser webcam
        cv2.imshow('OpenCV Feed', image)
        result_test = extract_keypoints(results)
        print(len(result_test))
        if cv2.waitKey(10) & 0xFF == ord('q'):  # Her sætter vi 'Q' som en stopklods for kameraet når det kører
            break
    cap.release()
    cv2.destroyAllWindows()