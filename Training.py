import os
import argparse
from detect import *
from model import *
from extract_datapoints import *
from training_gui import *

#path for exported data
data_path= os.path.join('MP_Data')

#actions we try to detect

ACTIONSDICT = {"Default": np.array(['A', 'B', 'C', 'D', 'E', 'Idle']), "Yubi-yay": np.array(['A', 'B', 'C', 'D', 'E', 'Idle', 'Kosovo'])}

ACTIONS = ACTIONSDICT["Default"]

#Determents how many frames of the video is used
DESIRED_LENGTH = 15

#Amount of epochs used when training
EPOCHS_AMOUNT = 2000

#The seed used when spliting train and test data
SEED = 1337

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and train a model based on a dataset with tensorflow")
    parser.add_argument("--extract", default=False, action="store_true")
    parser.add_argument("--gui", default=False, action="store_true")
    parser.add_argument("--face", default=False, action="store_true")
    parser.add_argument("--pose",default=False, action="store_true")
    parser.add_argument('--path',type=str, required=False, default='Training_videos', help='path for Training videos (default "Training_videos")')
    args = parser.parse_args()
    video_amount = count_videos(args.path, ACTIONS)

    if args.gui:
        #If the gui argument is present, launch gui instead of tui
        gui = Gui(ACTIONSDICT, data_path)
        gui.start_gui()
    else:
        #If you start the program with --extract behind it, it will extract data and train, if you start the program normally you only train.
        shape = 126
        if args.face:
            shape = shape + 1404
        if args.pose:
            shape = shape + 132
        
        if args.extract:
            extract_data(ACTIONS, video_amount, DESIRED_LENGTH, data_path, shape, args.path)

        model = YubiModel(DESIRED_LENGTH, shape, ACTIONS, data_path)
        model.train_model(EPOCHS_AMOUNT, video_amount, SEED)