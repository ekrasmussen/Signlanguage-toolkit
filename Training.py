import os
import argparse
from tkinter import filedialog
from detect import *
from model import *
from extract_datapoints import *
import tkinter as tk
from threading import Thread, Event

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

class Gui:

    def __init__(self):
        self.root = tk.Tk()
        self.stop_event = Event()
        self.stop_event.clear()
        self.thread = None

    def start_gui(self):
        labelx = 50
        spinboxx = 200
        self.root.geometry("400x300")
        self.root.title(f"Training & extract")

        self.root.resizable(0,0)
        canvas = tk.Canvas(self.root, width=400, height=300)
        canvas.pack()
        

        #Label that shows directory
        global label_directory_text
        label_directory_text = tk.StringVar()
        label_directory_text.set(f'Directory: {data_path}')
        label_directory = tk.Label(self.root, font=('Arial', 10), textvariable=label_directory_text)
        canvas.create_window(30, 230, window=label_directory, anchor=tk.W)

        #Button for selecting directory for training data
        directory_button = tk.Button(self.root, text="Select directory", font=('Arial', 10), command=self.select_directory)
        canvas.create_window(30, 260, window=directory_button, anchor=tk.W)

        #Extract checkbox
        #Used to get the value the check_box is in
        check_box_extract_value = tk.IntVar()
        check_box_extract = tk.Checkbutton(self.root, text='Extract & train', font=('Arial', 10), variable=check_box_extract_value, onvalue=1, offvalue=0)
        canvas.create_window(spinboxx, 180, window=check_box_extract, anchor=tk.W)

        #Actions dropdown
        label_actions = tk.Label(self.root, text="Action set:", font=('Arial', 10))
        canvas.create_window(labelx,60,window=label_actions, anchor=tk.W)

        dropdown_actions = ["Default", "Yubi-yay"]
        clicked = tk.StringVar()
        clicked.set(dropdown_actions[0])
        dropdown = tk.OptionMenu(self.root, clicked, *dropdown_actions)
        canvas.create_window(spinboxx, 60, window=dropdown, anchor=tk.W)

        #Desired length
        label_fps = tk.Label(self.root, text="Frame amount:", font=('Arial', 10))
        canvas.create_window(labelx,90,window=label_fps, anchor=tk.W)

        desired_length = tk.IntVar(value=1)
        spinbox_frames = tk.Spinbox(self.root,textvariable=desired_length, from_= 1, to = 90)
        canvas.create_window(spinboxx,90, window=spinbox_frames, anchor=tk.W)

        #Desired epochs
        label_epochs = tk.Label(self.root, text="Epochs:", font=('Arial', 10))
        canvas.create_window(labelx,120,window=label_epochs, anchor=tk.W)

        desired_epochs = tk.IntVar(value=1)
        spinbox_epochs = tk.Spinbox(self.root,textvariable=desired_epochs, from_= 1, to = 10000)
        canvas.create_window(spinboxx,120, window=spinbox_epochs, anchor=tk.W)

        #Desired seed
        label_seed = tk.Label(self.root, text="Seed:", font=('Arial', 10))
        canvas.create_window(labelx,150,window=label_seed, anchor=tk.W)

        desired_seed = tk.IntVar(value=1)
        spinbox_seed = tk.Spinbox(self.root,textvariable=desired_seed, from_= 1, to=100000)
        canvas.create_window(spinboxx,150, window=spinbox_seed, anchor=tk.W)
       
        #Button for starting the process assigned to thread
        button_start = tk.Button(self.root, text="Start", font=('Arial', 10), command=lambda : self.start_thread(check_box_extract_value, clicked, desired_length, desired_epochs, desired_seed))
        canvas.create_window(370, 260, window=button_start, anchor=tk.E)
        
        button_test = tk.Button(self.root, text="Test stop thread", font=('Arial', 10), command=lambda : self.stop_thread())
        canvas.create_window(370, 200, window=button_test, anchor=tk.E)

        self.root.mainloop()

    def execute_train(self, should_extract_data, clicked, frames, epochs, seed, stop_event):
        actions = ACTIONSDICT[clicked.get()]
        try:    
            videos = count_videos(actions)
            desired_length = frames.get()
            epochs_amount = epochs.get()
            seed = seed.get()

            if should_extract_data.get():
                extract_data(actions, videos, desired_length, data_path, stop_event)
            model = YubiModel(desired_length, SHAPE, actions, data_path)
            model.train_model(epochs_amount, videos, seed, stop_event)
        except:
            print('Check your values!')

    def select_directory(self):
        global data_path
        directory = filedialog.askdirectory(title="Select directory for training data")
        if directory:
            data_path = directory
            global label_directory_text
            label_directory_text.set(data_path)

    def start_thread(self, check_box_extract_value, clicked, desired_length, desired_epochs, desired_seed):
        self.stop_event.clear()
        
        #Create a Thread for the start method (to avoid hanging the gui when the training/extraction starts)
        self.thread = Thread(target=self.execute_train, args=(check_box_extract_value, clicked, desired_length, desired_epochs, desired_seed, self.stop_event))
        self.thread.start()
 

    def stop_thread(self):
        self.stop_event.set()
        self.thread.join()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and train a model based on a dataset with tensorflow")
    parser.add_argument("--extract", default=False, action="store_true")
    parser.add_argument("--gui", default=False, action="store_true")

    args = parser.parse_args()

    if args.gui:
        gui = Gui()
        gui.start_gui()
    else:
        #If you start the program with --extract behind it, it will extract data and train, if you start the program normally you only train.
        if args.extract:
            extract_data(ACTIONS, VIDEO_AMOUNT, DESIRED_LENGTH, data_path)
            
        model = YubiModel(DESIRED_LENGTH, SHAPE, ACTIONS, data_path)
        model.train_model(EPOCHS_AMOUNT, VIDEO_AMOUNT, SEED)