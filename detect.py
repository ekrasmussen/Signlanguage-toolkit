import cv2
import numpy as np


#used for detecting keypoints
def detect_keypoints(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = model.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return results

#used for extracting keypoints
def extract_all_keypoints(results, shape_size):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    result = []

    #If hands only
    if shape_size == 126:
        result = np.concatenate([lh, rh])
    
    #if pose and hands only
    elif shape_size == 258:
        result = np.concatenate([pose, lh, rh])

    #if face and hands only
    elif shape_size == 1530:
        result = np.concatenate([face, lh, rh])
    
    #If unknown value, return all
    else:
        result = np.concatenate([pose, face, lh, rh])
        
    return result

# #used for extracting keypoints
# def extract_hand_keypoints(results):
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#     return np.concatenate([lh, rh])
