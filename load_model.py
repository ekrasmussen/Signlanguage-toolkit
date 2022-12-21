import configparser
import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU


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
        #               Original Model inspired by Nicolas Renottes action recognition guide
        #               Source link: https://www.youtube.com/watch?v=doDUihpj6ro
        #               Nicolas' Github: https://github.com/nicknochnack
      
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.desired_length, self.shape)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))       #This line will have return_sequenecs set to False, because the next layer is a dense-layer so we need to not return the sequences to that layer
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.actions.shape[0], activation='softmax'))       #This line is going to return values that are within 0-1 and with the sums of all values returned adding up to 1, because of softmax
        model.summary()
        model.load_weights(f'{file_path}.h5')
        return model

    #Updates the sentence 
    def sentence_update(self, res): 
        print(f'res {res}')
        if np.unique(self.predictions[-10:])[0]==np.argmax(res): 
            if res[np.argmax(res)] > self.threshold: #res[np.argmax(res)] is grabbing the threshold. Checks whether or not res is above threshold 
                
                if len(self.sentence) > 0: #Checks if we have any sentence in variable sentence
                    if self.actions[np.argmax(res)] != self.sentence[-1]: #Checks that the predicted sign is not the same as the last one
                        self.sentence.append(self.actions[np.argmax(res)]) #Append the initial action to the sentence array. If there is no sentences in the current array, then the current action cant match whats in the sentence
                else:
                    self.sentence.append(self.actions[np.argmax(res)])

        if len(self.sentence) > 5: 
            self.sentence = self.sentence[-5:] #if len(self.sentence) is greater than 5, grab the last 5 values so we dont end up with giant array trying to render

    #Gives a prediction based on given keypoint.
    def predict(self, keypoints):
        self.sequence.append(keypoints) #Appending keypoints to sequence
        self.sequence = self.sequence[-self.desired_length:]
        
        is_desired_length = False #Is used outside of method to check if res is empty
        res = []
        if len(self.sequence) == self.desired_length: # If the length of self.squence is equal to self.desired_length, only then will there be made a prediction
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0] # the expand_dims function in NumPy is used to expand the shape of an input array that is passed to it. 
            print(self.actions[np.argmax(res)])
            self.predictions.append(np.argmax(res))
            is_desired_length = True

        return is_desired_length, res
            

        