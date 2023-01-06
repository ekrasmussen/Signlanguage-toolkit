from detect import *
from model import *
from extract_datapoints import *
import tkinter as tk
from tkinter import filedialog
from threading import Thread, Event

#21 keypoints * 2 hands * 3 scalars per keypoint (x, y, z)
KEYPOINTS_HANDS_AMOUNT = 126

#33 keypoints  * 4 scalars per keypoint (x, y, z, visibility)
KEYPOINTS_POSE_AMOUNT = 132

#468 * 3 scalars per keypoint
KEYPOINTS_FACE_AMOUNT = 1404

class Gui:

    def __init__(self, actions_dictionary, data_path):
        #Initialize a window
        self.root = tk.Tk()
        self.data_path = data_path
        self.videos_path = "Training_videos"
        self.actions_dictionary = actions_dictionary
        self.stop_event = Event()
        self.stop_event.clear()
        self.thread = None
    
    def start_gui(self):
        labelx = 50
        spinboxx = 200
        self.root.geometry("450x430")
        self.root.title(f"Training & extract")

        self.root.resizable(0,0)
        canvas = tk.Canvas(self.root, width=450, height=430)
        canvas.pack()

        #Label that shows directory
        global label_output_folder_text
        label_output_folder_text = tk.StringVar()
        label_output_folder_text.set(self.create_path_label(f'Extract to: {self.data_path}'))
        label_output_folder = tk.Label(self.root, font=('Arial', 10), textvariable=label_output_folder_text)
        canvas.create_window(450 - labelx, 325, window=label_output_folder, anchor=tk.E)

        #Button for selecting directory for training data
        button_output_folder = tk.Button(self.root, text="Select Extract Folder", font=('Arial', 10), command=self.select_output_directory)
        canvas.create_window(450 - labelx, 355, window=button_output_folder, anchor=tk.E)




        #Labels for Video Folder
        global label_videos_folder_text
        label_videos_folder_text = tk.StringVar()
        label_videos_folder_text.set(self.create_path_label(f'Videos: {self.videos_path}'))
        label_videos_folder = tk.Label(self.root, font=("Arial", 10), textvariable=label_videos_folder_text)
        canvas.create_window(labelx, 325, window=label_videos_folder, anchor=tk.W)

        button_videos_folder = tk.Button(self.root, text="Select Videos Folder", font=("Arial", 10), command=self.select_input_directory)
        canvas.create_window(labelx, 355, window=button_videos_folder, anchor=tk.W)


        #Extract checkbox
        #Used to get the value the checkbox is in
        checkbox_extract_value = tk.IntVar()
        checkbox_extract = tk.Checkbutton(self.root, text='Extract & train', font=('Arial', 10), variable=checkbox_extract_value, onvalue=1, offvalue=0)
        canvas.create_window(spinboxx, 180, window=checkbox_extract, anchor=tk.W)

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



        #Desired Length - Number of frames
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

        #Decide shape
        shape_pos_x = 85
        label_keypoints_y = 200
        label_keypoints = tk.Label(self.root, text="Keypoints:", font=('Arial', 10))

        checkbox_face_value = tk.IntVar()
        checkbox_pose_value = tk.IntVar()
        checkbox_hands_value = tk.IntVar()

        checkbox_face_shape = tk.Checkbutton(self.root, text='Face', font=('Arial', 10), variable=checkbox_face_value, onvalue=KEYPOINTS_FACE_AMOUNT, offvalue=0)
        checkbox_pose_shape = tk.Checkbutton(self.root, text='Pose', font=('Arial', 10), variable=checkbox_pose_value, onvalue=KEYPOINTS_POSE_AMOUNT, offvalue=0)
        checkbox_hands_shape = tk.Checkbutton(self.root, text='Hands', font=('Arial', 10), state='disabled', variable=checkbox_hands_value, onvalue=KEYPOINTS_HANDS_AMOUNT, offvalue=0)

        checkbox_hands_shape.select() # Makes sure the landmarks from hands are on as default

        canvas.create_window(shape_pos_x, 225, window=checkbox_face_shape, anchor=tk.W)
        canvas.create_window(shape_pos_x, 250, window=checkbox_pose_shape, anchor=tk.W)
        canvas.create_window(shape_pos_x, 275, window=checkbox_hands_shape, anchor=tk.W)
        canvas.create_window(labelx,label_keypoints_y,window=label_keypoints, anchor=tk.W)

        #Button for starting the process assigned to thread
        button_start = tk.Button(self.root, text="Start", font=('Arial', 10), command=lambda : self.start_thread(checkbox_extract_value, clicked, desired_length, desired_epochs, desired_seed, self.get_shape_size(checkbox_face_value, checkbox_pose_value, checkbox_hands_value)))
        canvas.create_window(450 / 2, 400, window=button_start, anchor=tk.CENTER)

        #In the event that user closes window, run x method
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    

        self.root.mainloop()
    
    #To limit the length of the paths when shown in labels
    def create_path_label(self,input):
        formatted_str = input
        
        #If path is longer than 25 we add dots at the end after 25 characters
        if len(formatted_str) > 25:
            formatted_str = formatted_str[0:25] + "..."
        
        return formatted_str
    
    def execute_train(self, should_extract_data, clicked, frames, epochs, seed, shape_size):
        actions = self.actions_dictionary[clicked.get()]
        try:    
            videos = count_videos(self.videos_path, actions)
            desired_length = frames.get()
            epochs_amount = epochs.get()
            seed = seed.get()

            if should_extract_data.get():
                extract_data(actions, videos, desired_length, self.data_path, shape_size, self.videos_path, self.stop_event)
            model = YubiModel(desired_length, shape_size, actions, self.data_path)
            model.train_model(epochs_amount, videos, seed, self.stop_event)
        except:
            print('Error! Something went wrong when counting videos/extracting.')
    
    def select_input_directory(self):
        directory = filedialog.askdirectory(title="Select directory for recorded videos")
        if directory:
            self.videos_path = directory
            global label_videos_folder_text
            label_videos_folder_text.set(self.create_path_label(f"Videos: {self.videos_path}"))


    def select_output_directory(self):
        directory = filedialog.askdirectory(title="Select directory for training data")
        if directory:
            self.data_path = directory
            global label_output_folder_text
            label_output_folder_text.set(self.create_path_label(f"Extract to: {self.data_path}"))
    
    def get_shape_size(self, checkbox_1, checkbox_2, checkbox_3):
        result = checkbox_1.get() + checkbox_2.get() + checkbox_3.get()
        return result

    #Function to start a thread, which also clears the stop event. If stop_event is set, functions dont run
    def start_thread(self, checkbox_extract_value, clicked, desired_length, desired_epochs, desired_seed, shape_size):
        self.stop_event.clear()
        print("Shape: " + str(shape_size))
        #Create a Thread for the start method (to avoid hanging the gui when the training/extraction starts)
        if 0 < desired_length.get() < 91 and 0 < desired_epochs.get() < 10001 and 0 < desired_seed.get() < 100001:
            self.thread = Thread(target=self.execute_train, args=(checkbox_extract_value, clicked, desired_length, desired_epochs, desired_seed, shape_size))
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