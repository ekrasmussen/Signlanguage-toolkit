from tkinter import filedialog
from detect import *
from model import *
from extract_datapoints import *
import tkinter as tk
from threading import Thread, Event

class Gui:

    def __init__(self, actions_dictionary, shape, data_path):
        #Initialize a window
        self.root = tk.Tk()
        self.data_path = data_path
        self.actions_dictionary = actions_dictionary
        self.shape = shape
        self.stop_event = Event()
        self.stop_event.clear()
        self.thread = None
    
    def start_gui(self):
        labelx = 50
        spinboxx = 200
        self.root.geometry("450x300")
        self.root.title(f"Training & extract")

        self.root.resizable(0,0)
        canvas = tk.Canvas(self.root, width=450, height=300)
        canvas.pack()

        #Label that shows directory
        global label_directory_text
        label_directory_text = tk.StringVar()
        label_directory_text.set(f'Directory: {self.data_path}')
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
        label_max_fps = tk.Label(self.root, text="Max: 90", font=("Arial", 10))
        
        canvas.create_window(labelx,90,window=label_fps, anchor=tk.W)
        canvas.create_window(labelx + 290,90,window=label_max_fps, anchor=tk.W)


        desired_length = tk.IntVar(value=1)
        spinbox_frames = tk.Spinbox(self.root,textvariable=desired_length, from_= 1, to = 90)
        canvas.create_window(spinboxx,90, window=spinbox_frames, anchor=tk.W)

        #Desired epochs
        label_epochs = tk.Label(self.root, text="Epochs:", font=('Arial', 10))
        label_max_epochs = tk.Label(self.root, text="Max: 10000", font=('Arial', 10))
        
        canvas.create_window(labelx,120,window=label_epochs, anchor=tk.W)
        canvas.create_window(labelx + 290, 120, window=label_max_epochs, anchor=tk.W)

        desired_epochs = tk.IntVar(value=1)
        spinbox_epochs = tk.Spinbox(self.root,textvariable=desired_epochs, from_= 1, to = 10000)
        canvas.create_window(spinboxx,120, window=spinbox_epochs, anchor=tk.W)

        #Desired seed
        label_seed = tk.Label(self.root, text="Seed:", font=('Arial', 10))
        label_max_seed = tk.Label(self.root, text="Max: 100000", font=('Arial', 10))
        
        canvas.create_window(labelx,150,window=label_seed, anchor=tk.W)
        canvas.create_window(labelx + 290,150,window=label_max_seed, anchor=tk.W)

        desired_seed = tk.IntVar(value=1)
        spinbox_seed = tk.Spinbox(self.root,textvariable=desired_seed, from_= 1, to=100000)
        canvas.create_window(spinboxx,150, window=spinbox_seed, anchor=tk.W)
       
        #Button for starting the process assigned to thread
        button_start = tk.Button(self.root, text="Start", font=('Arial', 10), command=lambda : self.start_thread(check_box_extract_value, clicked, desired_length, desired_epochs, desired_seed))
        canvas.create_window(370, 260, window=button_start, anchor=tk.E)

        #In the event that user closes window, run x method
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.root.mainloop()

    def execute_train(self, should_extract_data, clicked, frames, epochs, seed):
        actions = self.actions_dictionary[clicked.get()]
        try:    
            videos = count_videos(actions)
            desired_length = frames.get()
            epochs_amount = epochs.get()
            seed = seed.get()

            if should_extract_data.get():
                extract_data(actions, videos, desired_length, self.data_path, self.stop_event)
            model = YubiModel(desired_length, self.shape, actions, self.data_path)
            model.train_model(epochs_amount, videos, seed, self.stop_event)
        except:
            print('Error! Something went wrong.')

    def select_directory(self):
        directory = filedialog.askdirectory(title="Select directory for training data")
        if directory:
            self.data_path = directory
            global label_directory_text
            label_directory_text.set(self.data_path)

    #Function to start a thread, which also clears the stop event. If stop_event is set, functions dont run
    def start_thread(self, check_box_extract_value, clicked, desired_length, desired_epochs, desired_seed):
        self.stop_event.clear()
        
        #Create a Thread for the start method (to avoid hanging the gui when the training/extraction starts)
        if 0 < desired_length.get() < 91 and 0 < desired_epochs.get() < 10001 and 0 < desired_seed.get() < 100001:
            self.thread = Thread(target=self.execute_train, args=(check_box_extract_value, clicked, desired_length, desired_epochs, desired_seed))
            self.thread.start()
        else:
            raise ValueError('Check your values.')

 
    #Stops the thread and joins back on the main thread
    def stop_thread(self):
        self.stop_event.set()
        self.thread.join()

    #The on_closing function which is called when user closes window
    def on_closing(self):
        if self.thread is not None:
            self.stop_thread()
        self.root.destroy()
        