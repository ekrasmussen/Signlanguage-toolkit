import tkinter as tk
import mediapipe as mp
from datetime import datetime
import cv2
from PIL import Image, ImageTk
from detect import *
from load_model import *
from readerPrediction import mediapipe_detection

class Gui:

    #Constructor
    def __init__(self, desired_length, actions, file_path, x_res, y_res):
        self.root = tk.Tk()
        self.actions = actions
        self.x_res = x_res
        self.y_res = y_res
        self.cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[cv2.CAP_PROP_FRAME_WIDTH, x_res,
            cv2.CAP_PROP_FRAME_HEIGHT, y_res])
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.model = Model(desired_length, self.actions, file_path)
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (45, 117, 16), (117, 245, 16), (16, 117, 245)]
        #Make colors random or at least scale with more actions 

    #Setups the gui
    def setup_gui(self):

        self.root.geometry("800x600")
        self.root.title("ReaderGUI")

        canvas = tk.Canvas(self.root, width=800, height=600)
        canvas.pack()

        frame = tk.Frame(height=480, width=480)
        frame.place(x=300, y=300, anchor= tk.CENTER)
        global video_feed
        video_feed = tk.Label(frame)
        video_feed.place(x=0, y=0)
        global checkbox_display_landmarks_var
        checkbox_display_landmarks_var = tk.IntVar()

        checkbox_display_landmarks = tk.Checkbutton(self.root, text='Display landmarks', variable = checkbox_display_landmarks_var, font=('Arial', 10))
        canvas.create_window(300, 20, window = checkbox_display_landmarks)

        button_save_as_text = tk.Button(self.root, text='Save as text file', font= ('Arial', 10), command= self.save_to_text)
        canvas.create_window(600, 270, window= button_save_as_text, anchor= tk.NW)

    # Draws the signs and draws their probabilities
    def prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, f'{actions[num]}: {int(prob * 100)}%', (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return output_frame


    def viz(self, res, image):
        if np.unique(self.model.predictions[-10:])[0]==np.argmax(res): 
            if res[np.argmax(res)] > self.model.threshold: 
                
                if len(self.model.sentence) > 0: 
                    if self.actions[np.argmax(res)] != self.model.sentence[-1]:
                        self.model.sentence.append(self.actions[np.argmax(res)])
                else:
                    self.model.sentence.append(self.actions[np.argmax(res)])

        if len(self.model.sentence) > 5: 
            self.model.sentence = self.model.sentence[-5:]

        # Viz probabilities
        image = self.prob_viz(res, self.actions, image, self.colors)
        return image
    
    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks,
                                self.mp_holistic.FACEMESH_TESSELATION)  # Draws face landmarks
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                self.mp_holistic.POSE_CONNECTIONS)  # Draws pose landmarks
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                self.mp_holistic.HAND_CONNECTIONS)  # Draws left hand landmarks
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                self.mp_holistic.HAND_CONNECTIONS)  # Draws right hand landmarks


    def save_to_text(self):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        with open('Sentences.txt', 'a') as f:
            f.write(f' {dt_string}:{self.model.sentence} \n')
            f.close()

    def start(self):
        ret, frame = self.cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)

        if checkbox_display_landmarks_var.get(): # In python 0 is equal to false, and 1 is equal to true
            self.draw_landmarks(image, results)

        keypoints = extract_hand_keypoints(results)

        is_desired_length, res = self.model.predict(keypoints)

        if is_desired_length:
            image = self.viz(res, image)

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(self.model.sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #global sentence
        #image, sentence, predictions = start_read(keypoints, image, sequence, sentence, model)
        #display_confidence_text(predictions)

        #Setup for user webcam window
        ratio = self.x_res / self.y_res
        window_height = 480
        window_width = int(window_height * ratio)
        window_dimensions = (window_width, window_height)

        image = cv2.resize(image, window_dimensions)

        img = image
        imgarr = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(imgarr)
        video_feed.imgtk = imgtk
        video_feed.configure(image=imgtk)
        video_feed.after(10, self.start)

