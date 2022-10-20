import os
import argparse
from tkinter import filedialog
from detect import *
from model import *
from extract_datapoints import *
import tkinter as tk

#path for exported data
data_path= os.path.join('MP_Data')

#actions we try to detect

ACTIONSDICT = {"Default": np.array(['A', 'B', 'C', 'D', 'E', 'Idle']), "Yubi-yay": np.array(['A', 'B', 'C', 'D', 'E', 'Idle', 'Kosovo'])}

ACTIONS = ACTIONSDICT["Default"]

VIDEO_AMOUNT = count_videos(ACTIONS)

#Determents how many frames of the video is used
DESIRED_LENGTH = 15

#Input amount to the model, 126 inputs if using only hands
SHAPE = 126

#Amount of epochs used when training
EPOCHS_AMOUNT = 2000

#The seed used when spliting train and test data
SEED = 1337

def start(check_box_value, clicked, fps, epochs, seed):
    actions = ACTIONSDICT[clicked.get()]
    try:    
        videos = count_videos(actions)
        desired_length = fps.get()
        epochs_amount = epochs.get()
        seed = seed.get()

        if check_box_value.get():
            extract_data(actions, videos, desired_length, data_path)
        model = YubiModel(desired_length, SHAPE, actions, data_path)
        model.train_model(epochs_amount, videos, seed)
    except:
        print('Check your values!')

def select_directory():
    directory = filedialog.askdirectory(title="Select directory for training data")
    global data_path
    data_path = directory

def gui():
    root = tk.Tk()

    labelx = 100
    spinboxx = 275

    root.geometry("800x600")
    root.title(f"Training & extract")

    canvas = tk.Canvas(root, width=750, height=550)
    canvas.pack()

    #Button for starting the process
    start_button = tk.Button(root, text="Start", font=('Arial', 12), command=lambda : start(check_box_extract_value, clicked, desired_length, desired_epochs, desired_seed))
    canvas.create_window(650, 450, window=start_button)

    #Button for selecting directory for training data
    directory_button = tk.Button(root, text="Select directory", font=('Arial', 12), command=select_directory)
    canvas.create_window(500, 450, window=directory_button)

    #Extract checkbox
    #Used to get the value the check_box is in
    check_box_extract_value = tk.IntVar()
    check_box_extract = tk.Checkbutton(root, text='Extract & train', variable=check_box_extract_value, onvalue=1, offvalue=0)
    canvas.create_window(spinboxx, 200, window=check_box_extract)

    #Actions dropdown
    label_actions = tk.Label(root, text="Actions to extract and/or train")
    canvas.create_window(labelx,100,window=label_actions)
    dropdown_actions = ["Default", "Yubi-yay"]
    clicked = tk.StringVar()
    clicked.set(dropdown_actions[0])
    dropdown = tk.OptionMenu(root, clicked, *dropdown_actions)
    canvas.create_window(spinboxx,100, window=dropdown)

    #Desired length
    label_fps = tk.Label(root, text="Desired length/framerate")
    canvas.create_window(labelx,125,window=label_fps)
    desired_length = tk.IntVar(value=1)
    spinbox_fps = tk.Spinbox(root,textvariable=desired_length, from_= 1, to = 90)
    canvas.create_window(spinboxx,125, window=spinbox_fps)

    #Desired epochs
    label_epochs = tk.Label(root, text="Amount of epochs")
    canvas.create_window(labelx,150,window=label_epochs)
    desired_epochs = tk.IntVar(value=1)
    spinbox_epochs = tk.Spinbox(root,textvariable=desired_epochs, from_= 1, to = 10000)
    canvas.create_window(spinboxx,150, window=spinbox_epochs)

    #Desired seed
    label_seed = tk.Label(root, text="Select seed for training split")
    canvas.create_window(labelx,175,window=label_seed)
    desired_seed = tk.IntVar(value=1)
    spinbox_seed = tk.Spinbox(root,textvariable=desired_seed, from_= 1, to=100000)
    canvas.create_window(spinboxx,175, window=spinbox_seed)

    root.mainloop()


#Make start button call a funktion that will starts training, and extracting if chosen on check_box


# dropdown_options = ["Argh", "mah", ":)"]


# # datatype of the menu text, also get 
# clicked = tk.StringVar()
# # initial menu text
# clicked.set("Argh")

# dropdown = tk.OptionMenu(root, clicked, *dropdown_options)
# canvas.create_window(500, 300, window=dropdown)


# test_entry = tk.Entry(root, font=('Arial', 12))
# canvas.create_window(100, 100, window=test_entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and train a model based on a dataset with tensorflow")
    parser.add_argument("--extract", default=False, action="store_true")
    parser.add_argument("--gui", default=False, action="store_true")

    args = parser.parse_args()

    if args.gui:
        gui()
    else:
        #If you start the program with --extract behind it, it will extract data and train, if you start the program normally you only train.
        if args.extract:
            extract_data(ACTIONS, VIDEO_AMOUNT, DESIRED_LENGTH, DATA_PATH)
            
        model = YubiModel(DESIRED_LENGTH, SHAPE, ACTIONS, DATA_PATH)
        model.train_model(EPOCHS_AMOUNT, VIDEO_AMOUNT, SEED)