from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from threading import Event
import configparser

class YubiModel:

    #constructor
    def __init__(self, desired_length, shape, actions, data_path, is_test=False):
        self.desired_length = desired_length
        self.shape = shape 
        self.actions = actions
        self.data_path = data_path
        self.model = self.create_model()
        self.is_test = is_test
        self.create_log()
        self.logs_path, self.timestamp = self.make_logs_path()
        self.n_epochs = 0
       
    def create_model(self):
        #               Original Model inspired by Nicholas Renottes action recognition guide
        #               Source link: https://www.youtube.com/watch?v=doDUihpj6ro
        #               Nicholas' Github: https://github.com/nicknochnack
        
        #sets up sequential layers in neural network
        model = Sequential()
        #adds a LSTM layer with 64 nodes, returns a sequence, uses relu activation, and input shape
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.desired_length, self.shape)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        #note this doesn't return a sequence
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        #adds dense layer with 64 nodes uses relu activation
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation="relu"))
        #activate returns values with a probability between 0.0 and 1.0 and sum of the values adds up to 1
        model.add(Dense(self.actions.shape[0], activation='softmax'))

        #compiles model using categorical_crossentropy due to multiple features being used
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        return model

    def make_logs_path(self):
        #sets current date and time for file naming
        now = datetime.now()

        #formats timestamp
        timestamp = now.strftime("%d-%m-%Y %H-%M-%S")
        
        #creates the folder with corresponding timestamp inside the logs folder
        logs_path = ""
        if not self.is_test:
            logs_path = os.path.join('Logs', timestamp)
        else:
            logs_path = os.path.join('Logs', f"{timestamp}_test")
        
        try:
            os.makedirs(logs_path)
        except:
            print(f"Folder already named {timestamp}, files inside will be overwritten")

        return logs_path, timestamp


    def create_log(self):
        #create logs folder if it doesnt exist
        try:
            os.makedirs("Logs")
        except:
            print("Logs folder already exists. Skipping..")
        
    
    def save_confusion_matrix(self, confusion_matrix):
        #convert and save confusion matrix individually by action
        with pd.ExcelWriter(os.path.join(f"{self.logs_path}", "confusion_matrix.xlsx")) as writer:
            for matrix in range(0, len(confusion_matrix)):
                cmdf = pd.DataFrame([confusion_matrix[matrix][0], confusion_matrix[matrix][1]], index=['Positive','Negative'], columns=['Positive','Negative'])
                cmdf.to_excel(writer, sheet_name=self.actions[matrix])

    def save_training_info(self, accuracy, seed):
        #saves accuracy score as a simple txt file
        
        file = open(os.path.join(f"{self.logs_path}", "training_info.txt"), "w+")
        file.write(f"Accuracy: {accuracy}\nDesired length: {self.desired_length}\nEpochs: {self.n_epochs}\nSeed: {seed}")
        file.close()
    
    def train_model(self, epochs_amount, videoAmount, seed, stop_event = Event()):
        if not stop_event.is_set():
            #maps labels to numbers in a dictionary
            label_map = {label:num for num, label in enumerate(self.actions)}

            #creates two lists
            sequences, labels = [], []
            i = 0
            #goes through actions
            for action in self.actions:
                
                no_sequences = videoAmount[i]

                #goes through sequences
                for sequence in range(no_sequences):
                    #creates list of frames for a sequence
                    window = []
                    #goes through sequences
                    for frame_num in range(self.desired_length):
                        #loads numpy array
                        res = np.load(os.path.join(self.data_path, action, str(sequence), "{}.npy".format(frame_num)))
                        #appends frame to window list
                        window.append(res)
                    #appends window list to sequences list
                    sequences.append(window)
                    #appends label map for action to labels list
                    labels.append(label_map[action])

                i += 1
            
            #set X for model training
            X = np.array(sequences)
            #set y for model training
            y = to_categorical(labels).astype(int)

            #splits dataset in training and test sets
            #random_state sets seed value to allow for comparison of different neural networks
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            #sets condition for earlystopping
            #monitor loss
            #mode is set to min to stop when loss stops descreasing 
            #patience sets number of epochs after which training will be stopped
            #restore_best_weights restores best weights after stopping
            earlystopping = EarlyStopping(monitor = 'loss', min_delta = 0,  patience = 20, verbose = 0, mode = 'auto', baseline = None, restore_best_weights = True)

            #trains the model
            #do not specify the batch_size if your data is in the form of a dataset, generators, or keras.utils.Sequence instances
            m = self.model.fit(X_train, y_train, epochs = epochs_amount, callbacks=earlystopping)

            #saves model
            self.model.save(f'{self.timestamp}.h5')

            #sets yhat from prediction on X_test
            yhat = self.model.predict(X_test)

            ytrue = np.argmax(y_test, axis=1).tolist()
            yhat = np.argmax(yhat, axis=1).tolist()

            #creates and saves confusion matrix
            confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
            self.save_confusion_matrix(confusion_matrix)

            #creates and saves training info
            accuracy = accuracy_score(ytrue, yhat)
            self.n_epochs = len(m.history['loss'])
            self.save_training_info(accuracy, seed)
            self.save_as_config_file()

    def save_as_config_file(self):
        config = configparser.ConfigParser()
        config.add_section('Model')

        config['Model']['Length'] = str(self.desired_length)
        config['Model']['Shape'] = str(self.shape)
        config['Model']['Actions set'] = str(self.actions)

        with open(f'{self.timestamp}.ini', 'w') as configfile:
            config.write(configfile)