from cProfile import label
from msilib.schema import CheckBox
import tkinter as tk
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from readerPrediction import *
from threading import Thread
from datetime import datetime

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def gui():
    global window
    window = tk.Tk()

    window.geometry("800x600")
    window.title("ReaderGUI")

    canvas = tk.Canvas(window, width=800, height=600)
    canvas.pack()

    frame = tk.Frame(height=480, width=480)
    frame.place(x=300, y=300, anchor= tk.CENTER)
    global video_feed
    video_feed = tk.Label(frame)
    video_feed.place(x=0, y=0)
    global checkbox_display_landmarks_var
    checkbox_display_landmarks_var = tk.IntVar()

    checkbox_display_landmarks = tk.Checkbutton(window, text='Display landmarks', variable = checkbox_display_landmarks_var, font=('Arial', 10))
    canvas.create_window(300, 20, window = checkbox_display_landmarks)

    label_a_value = tk.StringVar()
    label_b_value = tk.StringVar()
    label_c_value = tk.StringVar()
    label_d_value = tk.StringVar()
    label_e_value = tk.StringVar()
    label_idle_value = tk.StringVar()
    global prop_labels
    prop_labels = [label_a_value, label_b_value, label_c_value, label_d_value, label_e_value, label_idle_value]
    labelx = 600
    label_confidence = tk.Label(window, text='Confidence Score', font=('Arial', 10))
    label_a = tk.Label(window, text='A', font=('Arial', 10), textvariable = label_a_value)
    label_b = tk.Label(window, text='B', font=('Arial', 10), textvariable = label_b_value)
    label_c = tk.Label(window, text='c', font=('Arial', 10), textvariable = label_c_value)
    label_d = tk.Label(window, text='D', font=('Arial', 10), textvariable = label_d_value)
    label_e = tk.Label(window, text='E', font=('Arial', 10), textvariable = label_e_value)
    label_idle = tk.Label(window, text='Idle', font=('Arial', 10), textvariable = label_idle_value)
    canvas.create_window(labelx, 60, window= label_confidence, anchor= tk.NW)
    canvas.create_window(labelx, 90, window = label_a, anchor = tk.NW)
    canvas.create_window(labelx, 120, window=label_b, anchor=tk.NW)
    canvas.create_window(labelx, 150, window=label_c, anchor=tk.NW)
    canvas.create_window(labelx, 180, window=label_d, anchor=tk.NW)
    canvas.create_window(labelx, 210, window=label_e, anchor=tk.NW)
    canvas.create_window(labelx, 240, window=label_idle, anchor=tk.NW)

    button_save_as_text = tk.Button(window, text='Save as text file', font= ('Arial', 10), command= save_to_text)
    canvas.create_window(labelx, 270, window= button_save_as_text, anchor= tk.NW)


cap = cv2.VideoCapture(0)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
sequence = []
sentence = [] #Sentence is the last 5 known / guessed signs displayed at the top of the image
model = load_model()

def display_confidence_text(predictions):

    if predictions:
        for num, prop in enumerate(predictions):
            prop_labels[num].set(f'{actions[num]}: {prop}%')

def start():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if checkbox_display_landmarks_var.get(): # In python 0 is equal to false, and 1 is equal to true
            draw_landmarks(image, results)

        keypoints = extract_keypoints_hands(results)

        global sentence
        image, sentence, predictions = start_read(keypoints, image, sequence, sentence, model)
        display_confidence_text(predictions)

        img = image
        imgarr = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(imgarr)
        video_feed.imgtk = imgtk
        video_feed.configure(image=imgtk)
        video_feed.after(10, start)

def save_to_text():
    global sentence
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open('Sentences.txt', 'a') as f:
        f.write(f' {dt_string}:{sentence} \n')
        f.close()

gui()
start()
window.mainloop()