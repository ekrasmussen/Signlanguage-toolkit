import tkinter as tk
import mediapipe as mp
from datetime import datetime
import cv2
from PIL import Image, ImageTk
from detect import *
from load_model import *

class Gui:

    #Constructor
    def __init__(self, desired_length, actions, file_path, x_res, y_res, display_amount):
        self.root = tk.Tk()
        self.actions = actions
        self.x_res = x_res
        self.y_res = y_res
        self.display_amount = display_amount 
        self.cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[cv2.CAP_PROP_FRAME_WIDTH, x_res,
            cv2.CAP_PROP_FRAME_HEIGHT, y_res])
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.model = Model(desired_length, self.actions, file_path)

    #Setups the gui
    def setup_gui(self):

        self.root.geometry("760x570") #Size of the window
        self.root.title("ReaderGUI") #Window Title

        canvas = tk.Canvas(self.root, width=760, height=570)
        canvas.pack()

        #frame holds the image
        frame = tk.Frame(width=640, height=480)
        frame.place(x=380, y=260, anchor= tk.CENTER)
        global video_feed, checkbox_display_landmarks_var
        video_feed = tk.Label(frame)
        video_feed.place(x=0, y=0)
        checkbox_display_landmarks_var = tk.IntVar()

        checkbox_display_landmarks = tk.Checkbutton(self.root, text='Display landmarks', variable = checkbox_display_landmarks_var, font=('Arial', 10))
        canvas.create_window(60, 530, window = checkbox_display_landmarks, anchor= tk.W)

        button_save_as_text = tk.Button(self.root, text='Save as text file', font= ('Arial', 10), command= self.save_to_text)
        canvas.create_window(700, 530, window= button_save_as_text, anchor= tk.E)

    # Draws the signs and draws their probabilities
    def prob_viz(self, res, input_frame):
        action_prob = []
        #Enumerates over the results to create a 2d list of actions index and their confidence score
        for num, prob in enumerate(res):
            action_prob.append([num, prob])

        #Sorts the 2d list decending in value
        sorted_action_prob = sorted(action_prob, key = lambda i: i[1], reverse=True)
        output_frame = input_frame.copy()

        #A loop that puts the 5 highest displayed on screen 
        for x in range(0, self.display_amount):
            cv2.rectangle(output_frame, (0, 60 + x * 40), (int(sorted_action_prob[x][1] * 100), 90 + x * 40), (0, 200, 93), -1)
            cv2.putText(output_frame, f'{self.actions[sorted_action_prob[x][0]]}: {int(sorted_action_prob[x][1] * 100)}%', (0, 85 + x * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return output_frame

    #Draw landmarks
    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks,
                                self.mp_holistic.FACEMESH_TESSELATION)  # Draws face landmarks
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                self.mp_holistic.POSE_CONNECTIONS)  # Draws pose landmarks
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                self.mp_holistic.HAND_CONNECTIONS)  # Draws left hand landmarks
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                self.mp_holistic.HAND_CONNECTIONS)  # Draws right hand landmarks

    #Save the current sentence in text 
    def save_to_text(self):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        with open('Sentences\Sentences.txt', 'a') as f:
            f.write(f'{dt_string}:{self.model.sentence} \n')
            f.close()
   
    #Draws sentence on image
    def draw_sentence(self, image):
        cv2.rectangle(image, (0,0), (640, 40), (0, 150, 255), -1)
        cv2.putText(image, ' '.join(self.model.sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    #Start showing cam, cam overlay, and predict sign
    def start(self):
        ret, frame = self.cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Scales image to fit the frame
        ratio = self.x_res / self.y_res
        window_height = 480
        window_width = int(window_height * ratio)
        window_dimensions = (window_width, window_height)

        image = cv2.resize(image, window_dimensions)

        results = self.holistic.process(image)

        if checkbox_display_landmarks_var.get(): # In python 0 is equal to false, and 1 is equal to true
            self.draw_landmarks(image, results)

        keypoints = extract_hand_keypoints(results)

        is_desired_length, res = self.model.predict(keypoints)

        #Checks if res is not empty
        if is_desired_length:
            self.model.sentence_update(res)
            image = self.prob_viz(res, image)

        image = self.draw_sentence(image)

        #Displays the camera on gui
        img = image
        imgarr = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(imgarr)
        video_feed.imgtk = imgtk
        video_feed.configure(image=imgtk)
        video_feed.after(10, self.start)
        self.root.mainloop()
