import os
import argparse
from detect import *
from model import *
from extract_datapoints import *
from training_gui import *

#path for exported data
data_path = os.path.join('MP_Data')

#actions we try to detect
ACTIONSDICT = {"Default": np.array(['A', 'B', 'C', 'D', 'E', 'Idle']), "Yubi-yay": np.array(['A', 'B', 'C', 'D', 'E', 'Idle','Dropper', 'Hue', 'Hvor', 'Jubilæum', 'Sejr'])}

def check_values(desired_length, seed, epochs_amount, actionset):
     
    failed = False
    actions = np.array([])

    if not 0 < desired_length < 91:
        failed = True
        print('frames can only be between 1 to 90')
    
    if not 0 < seed < 100001:
        failed = True
        print('seed can only be between 1 to 100000')
    
    if not 0 < epochs_amount < 10001:
        failed = True
        print('epochs can only be between 1 to 10000')

    actionset = actionset.capitalize()
    if actionset == "Yubi-yay":
        actions = ACTIONSDICT["Yubi-yay"]
    elif actionset == "Default":
        actions = ACTIONSDICT["Default"]
    else:
        failed = True
        print(f'actionset is not valid. Valid options are {list(ACTIONSDICT.keys())}')
    return failed, actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and train a model based on a dataset with tensorflow")
    parser.add_argument("--extract", default=False, action="store_true")
    parser.add_argument("--gui", default=False, action="store_true")
    parser.add_argument("--face", default=False, action="store_true")
    parser.add_argument("--pose",default=False, action="store_true")
    parser.add_argument("--frames", type=int, required=False, default=1, help='Amount of frames that are to be extracted from videos (default 1)')
    parser.add_argument("--epochs", type=int, required=False, default=1, help='Amount of epochs used for training (default 1)')
    parser.add_argument("--seed", type=int, required=False, default=1, help='The seed used to split test and training data (default 1)')
    parser.add_argument('--path',type=str, required=False, default='Training_videos', help='path for Training videos (default "Training_videos")')
    parser.add_argument("--actionset",type=str, required=False, default="Default", help='Choose actions set (default "default")')
    args = parser.parse_args()

    if args.gui:
        #If the gui argument is present, launch gui instead of tui
        gui = Gui(ACTIONSDICT, data_path)
        gui.start_gui()
    else:
        #Sets up variables from params
        actionset = args.actionset
        seed = args.seed
        desired_length = args.frames
        epochs_amount = args.epochs
        path = args.path

        failed, actions = check_values(desired_length, seed, epochs_amount, actionset)
    
        if not failed: 
            video_amount = count_videos(path, actions)
            #If you start the program with --extract behind it, it will extract data and train, if you start the program normally you only train.
            shape = 126
            if args.face:
                shape = shape + 1404
            if args.pose:
                shape = shape + 132
            
            if args.extract:
                extract_data(actions, video_amount, desired_length, data_path, shape, path)

            model = YubiModel(desired_length, shape, actions, data_path)
            model.train_model(epochs_amount, video_amount, seed)
        else:
            print('Not started invalid values check fail message above')