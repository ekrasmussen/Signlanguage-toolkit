import configparser
import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class Model:
    
    #Constructor
    def __init__(self, file_path):
        self.desired_length = 0
        self.shape = 0
        self.actions = np.array(0)
        self.read_config(file_path)
        self.model = self.load_model(file_path)
        self.sequence = []
        self.sentence = [] 
        self.predictions = []
        self.threshold = 0.8

    #Reads
    def read_config(self, file_path):
        config = configparser.ConfigParser()
        config.read(f'{file_path}.ini')
        self.desired_length = int(config['Model']['Length'])
        self.shape = int(config['Model']['Shape'])
        actions_string = config['Model']['Actions set']
        self.actions = np.array(re.sub('[^A-Za-z ]+', '', actions_string).split())

    #Creates/loads model. file_path is the path to the h5 file 
    def load_model(self, file_path):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.desired_length, self.shape)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.actions.shape[0], activation='softmax'))
        model.summary()
        model.load_weights(f'{file_path}.h5')
        return model

    #Updates the sentence 
    def sentence_update(self, res): 
        print(f'res {res}')
        if np.unique(self.predictions[-10:])[0]==np.argmax(res): 
            if res[np.argmax(res)] > self.threshold: 
                
                if len(self.sentence) > 0: 
                    if self.actions[np.argmax(res)] != self.sentence[-1]: #Checks that the predicted sign is not the same as the last one
                        self.sentence.append(self.actions[np.argmax(res)])
                else:
                    self.sentence.append(self.actions[np.argmax(res)])

        if len(self.sentence) > 5: 
            self.sentence = self.sentence[-5:]

    #Gives a prediction based on given keypoint.
    def predict(self, keypoints):
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-self.desired_length:]
        
        is_desired_length = False #Is used outside of method to check if res is empty
        res = []
        if len(self.sequence) == self.desired_length:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            print(self.actions[np.argmax(res)])
            self.predictions.append(np.argmax(res))
            is_desired_length = True

        return is_desired_length, res
            

        