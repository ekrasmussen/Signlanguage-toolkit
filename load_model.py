import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class Model:

    def __init__(self, desired_length, actions, file_path):
        self.desired_length = desired_length
        self.actions = actions
        self.model = self.load_model(file_path)
        self.sequence = []
        self.sentence = [] #Maybe move sentence to Gui
        self.predictions = []
        self.threshold = 0.8

    #file_path is the path to the h5 file 
    def load_model(self, file_path):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.desired_length, 126)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.actions.shape[0], activation='softmax'))
        model.summary()
        model.load_weights(file_path)
        return model


    def predict(self, keypoints):
        # 2. Prediction logic
        self.sequence.append(keypoints)
        self.sequence = self.sequence[- self.desired_length:]
        
        is_desired_length = False
        res = []

        if len(self.sequence) == self.desired_length:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            print(self.actions[np.argmax(res)])
            self.predictions.append(np.argmax(res))
            is_desired_length = True

        return is_desired_length, res
            

        