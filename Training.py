import os
import argparse
import tkinter
from detect import *
from model import *
from extract_datapoints import *
import tkinter as tk

#path for exported data
DATA_PATH = os.path.join('MP_Data')

#actions we try to detect
ACTIONS = np.array(['A', 'B', 'C', 'D', 'E', 'Idle']) 

#VIDEO_AMOUNT = count_videos(ACTIONS)

#Determents how many frames of the video is used
DESIRED_LENGTH = 15

#Input amount to the model, 126 inputs if using only hands
SHAPE = 126

#Amount of epochs used when training
EPOCHS_AMOUNT = 2000

#The seed used when spliting train and test data
SEED = 1337


def test_2():
    print(check_box_value.get())

root = tk.Tk()

root.geometry("800x600")
root.title(f"Training & extract")

canvas = tk.Canvas(root, width=750, height=550)
canvas.pack()

start_button = tk.Button(root, text="Start", font=('Arial', 12), command=test_2)
canvas.create_window(650, 450, window=start_button)

#Used to get the value the check_box is in
check_box_value = tk.IntVar()

check_box = tk.Checkbutton(root, text='Extract & train', variable=check_box_value, onvalue=1, offvalue=0)

canvas.create_window(200, 200, window=check_box)


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




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Create and train a model based on a dataset with tensorflow")
#     parser.add_argument("--extract", default=False, action="store_true")

#     args = parser.parse_args()

#     #If you start the program with --extract behind it, it will extract data and train, if you start the program normally you only train.
#     if args.extract:
#         extract_data(ACTIONS, VIDEO_AMOUNT, DESIRED_LENGTH, DATA_PATH)
#         print("extracting data too")
        
#     model = YubiModel(DESIRED_LENGTH, SHAPE, ACTIONS, DATA_PATH)
#     model.train_model(EPOCHS_AMOUNT, VIDEO_AMOUNT, SEED)