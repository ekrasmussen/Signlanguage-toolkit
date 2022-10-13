import os
import argparse
from detect import *
from model import *
from extract_datapoints import *

#from keras.callbacks import TensorBoard, callbacks

#path for exported data
DATA_PATH = os.path.join('MP_Data')

#actions we try to detect
ACTIONS = np.array(['A', 'B', 'C', 'D', 'E', 'Idle']) 

VIDEO_AMOUNT = count_videos(ACTIONS)

#Determents how many frames of the video is used
DESIRED_LENGTH = 15

#Input amount to the model, 126 inputs if using only hands
SHAPE = 126

#Amount of epochs used when training
EPOCHS_AMOUNT = 150

#The seed used when spliting train and test data
SEED = 1337


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and train a model based on a dataset with tensorflow")
    parser.add_argument("--extract", default=False, action="store_true")

    args = parser.parse_args()

    #If you start the program with --extract behind it, it will extract data and train, if you start the program normally you only train.
    if args.extract:
        extract_data(ACTIONS, VIDEO_AMOUNT, DESIRED_LENGTH, DATA_PATH)
        print("extracting data too")
        
    model = YubiModel(DESIRED_LENGTH, SHAPE, ACTIONS, DATA_PATH)
    model.train_model(EPOCHS_AMOUNT, VIDEO_AMOUNT, SEED)
    
