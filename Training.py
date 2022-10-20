import os
import argparse
from tkinter import filedialog
from detect import *
from model import *
from extract_datapoints import *
import tkinter as tk
from threading import Thread

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

def start(check_box_value, clicked, frames, epochs, seed):
    actions = ACTIONSDICT[clicked.get()]
    try:    
        videos = count_videos(actions)
        desired_length = frames.get()
        epochs_amount = epochs.get()
        seed = seed.get()

        if check_box_value.get():
            extract_data(actions, videos, desired_length, data_path)
        model = YubiModel(desired_length, SHAPE, actions, data_path)
        model.train_model(epochs_amount, videos, seed)
    except:
        print('Check your values!')

def select_directory():
    global data_path
    directory = filedialog.askdirectory(title="Select directory for training data")
    if directory:
        data_path = directory
        global label_directory_text
        label_directory_text.set(data_path)


def gui():
    root = tk.Tk()
    

    labelx = 50
    spinboxx = 200

    root.geometry("400x300")
    root.title(f"Training & extract")

    canvas = tk.Canvas(root, width=400, height=300)
    canvas.pack()
    

    #Label that shows directory
    global label_directory_text
    label_directory_text = tk.StringVar()
    label_directory_text.set(f'Directory: {data_path}')
    label_directory = tk.Label(root, font=('Arial', 10), textvariable=label_directory_text)
    canvas.create_window(30, 230, window=label_directory, anchor=tk.W)

    #Button for selecting directory for training data
    directory_button = tk.Button(root, text="Select directory", font=('Arial', 10), command=select_directory)
    canvas.create_window(30, 260, window=directory_button, anchor=tk.W)

    #Extract checkbox
    #Used to get the value the check_box is in
    check_box_extract_value = tk.IntVar()
    check_box_extract = tk.Checkbutton(root, text='Extract & train', font=('Arial', 10), variable=check_box_extract_value, onvalue=1, offvalue=0)
    canvas.create_window(spinboxx, 180, window=check_box_extract, anchor=tk.W)

    #Actions dropdown
    label_actions = tk.Label(root, text="Action set:", font=('Arial', 10))
    canvas.create_window(labelx,60,window=label_actions, anchor=tk.W)

    dropdown_actions = ["Default", "Yubi-yay"]
    clicked = tk.StringVar()
    clicked.set(dropdown_actions[0])
    dropdown = tk.OptionMenu(root, clicked, *dropdown_actions)
    canvas.create_window(spinboxx, 60, window=dropdown, anchor=tk.W)

    #Desired length
    label_fps = tk.Label(root, text="Frame amount:", font=('Arial', 10))
    canvas.create_window(labelx,90,window=label_fps, anchor=tk.W)

    desired_length = tk.IntVar(value=1)
    spinbox_frames = tk.Spinbox(root,textvariable=desired_length, from_= 1, to = 90)
    canvas.create_window(spinboxx,90, window=spinbox_frames, anchor=tk.W)

    #Desired epochs
    label_epochs = tk.Label(root, text="Epochs:", font=('Arial', 10))
    canvas.create_window(labelx,120,window=label_epochs, anchor=tk.W)

    desired_epochs = tk.IntVar(value=1)
    spinbox_epochs = tk.Spinbox(root,textvariable=desired_epochs, from_= 1, to = 10000)
    canvas.create_window(spinboxx,120, window=spinbox_epochs, anchor=tk.W)

    #Desired seed
    label_seed = tk.Label(root, text="Seed:", font=('Arial', 10))
    canvas.create_window(labelx,150,window=label_seed, anchor=tk.W)

    desired_seed = tk.IntVar(value=1)
    spinbox_seed = tk.Spinbox(root,textvariable=desired_seed, from_= 1, to=100000)
    canvas.create_window(spinboxx,150, window=spinbox_seed, anchor=tk.W)

    #Create a Thread for the start method (to avoid hanging the gui when the training/extraction starts)
    thread = Thread(target=start, args=(check_box_extract_value, clicked, desired_length, desired_epochs, desired_seed))
    
    #Button for starting the process assigned to thread
    start_button = tk.Button(root, text="Start", font=('Arial', 10), command=lambda : thread.start())
    canvas.create_window(370, 260, window=start_button, anchor=tk.E)
    
    root.mainloop()


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
            extract_data(ACTIONS, VIDEO_AMOUNT, DESIRED_LENGTH, data_path)
            
        model = YubiModel(DESIRED_LENGTH, SHAPE, ACTIONS, data_path)
        model.train_model(EPOCHS_AMOUNT, VIDEO_AMOUNT, SEED)