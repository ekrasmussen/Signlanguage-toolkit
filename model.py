from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from datetime import datetime


class YubiModel:

    
    def __init__(self, desired_length, shape, actions, data_path):
        self.desired_length = desired_length
        self.shape = shape 
        self.actions = actions
        self.data_path = data_path
        self.model = self.create_model()
       
    
    def create_model(self):
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

    def train_model(self, epochs_amount, videoAmount):
        #maps labels to numbers
        label_map = {label:num for num, label in enumerate(self.actions)}

        #creates to lists
        sequences, labels = [], []
        i = 0
        #goes through actions
        for action in self.actions:
            
            no_sequences = videoAmount[i]

            #goes through sequences
            for sequence in range(no_sequences):
                #creates list for of frames for a sequence
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

        #sets path for log file
        log_dir = os.path.join('Logs')
        #sets condition for earlystopping
        #monitor val_loss
        #mode is set to min to stop when val_loss stops descreasing 
        #patience sets number of epochs after which training will be stopped
        #restore_best_weights restores best weights after stopping
        #earlystopping = callbacks.Earlystopping(monitor = 'val_loss', mode = 'min', patience = 5, restore_best_weights = True)

        #trains the model
        #do not specify the batch_size if your data is in the form of a dataset, generators, or keras.utils.Sequence instances
        self.model.fit(X_train, y_train, epochs = epochs_amount)

        #sets currentDateTime for file naming
        currentDateTime = datetime.now()
        timestamp = datetime.timestamp(currentDateTime)
        #saves model
        self.model.save(f'{timestamp}_action.h5')

        #sets yhat from prediction on X_test
        yhat = self.model.predict(X_test)

        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()

        #creates confusion matrix
            #confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
        #saves confusion matrix
        #confusion_matrix.to_csv(f'{timestamp}_confusion_matrix')

        #creates accuracy score
        accuracy = accuracy_score(ytrue, yhat)
        #saves accuracy score
        #accuracy.to_csv(f'{timestamp}_accuracy')


    

   
