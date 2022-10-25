import argparse
from turtle import window_height, window_width
from typing_extensions import Required

import cv2
import os
import sys

def setup():
    #Arguments for running the software, all values have a default value and can be changed
    #Only required argument is the labels
    parser = argparse.ArgumentParser(description="Record videos intended to be part of dataset")
    parser.add_argument('--output', type=str, required=False, default="Recordings", help='Name of folder for outputs (Default "Recordings")')
    parser.add_argument('--recordtime', type=int, required=False, default=2, help="Amount of time to record in seconds.(Default 2)")
    parser.add_argument('--breaktime', type=int, required=False, default=3, help="Amount of break time between recordings in seconds.(Default 3)")
    parser.add_argument('--fps', type=int, required=False, default=30, help="Cameras FPS (Default 30)")
    parser.add_argument('Labels', metavar='Label', type=str, nargs='+', help="Labels for the recording session.")
    parser.add_argument('--amount', type=int, required=False, default=1, help="Amount of videos to record for each label")
    parser.add_argument('--xres', type=int, required=False, default=640, help="Width of camera resolution. (Default 640)")
    parser.add_argument('--yres', type=int, required=False, default=480, help="Height of camera resolution. (Default 480)")
    args = parser.parse_args() 

    return args

#Creates output folder for recording
def create_recordings_folder(output_folder_name):
    try:
        os.makedirs(output_folder_name)
    except:
        pass

#Creates labels for videos in output folder
def create_labels(output_folder_name, labels):
    for label in labels:
        try:
            os.makedirs(f"{output_folder_name}/{label}")
        except:
            pass

#Records webcam
def webcam_record(labels, fps, recordtime, breaktime, amount, output):
    camera = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[cv2.CAP_PROP_FRAME_WIDTH, camera_width,
    cv2.CAP_PROP_FRAME_HEIGHT, camera_height])

    #Setup for user webcam window
    ratio = camera_width / camera_height
    window_height = 480
    window_width = int(window_height * ratio)
    window_dimensions = (window_width, window_height)

    if camera.isOpened:
        #Loops through labels in list
        for label in labels:
            #Creates a new video 
            for video_number in range(amount):
                #VideoWriter writes to storage
                #Uses parameters: path for videos, video codec, frame rate, and image dimensions
                result = cv2.VideoWriter(f"{output}/{label}/{video_number}.mp4", cv2.VideoWriter.fourcc('m','p','4','v'), fps, (camera_width, camera_height))
                #Loops through frames of video
                for frame in range(recordtime + breaktime):
                    #Gets boolean and frame
                    ret, image = camera.read()
                    if ret:
                        #Resize image for better UX
                        resized_image = cv2.resize(image, window_dimensions)
                        #If anything is captured
                        #Save the image before processing with graphics
                        if frame > breaktime:
                            result.write(image)
                            #Graphics for recording
                            resized_image = cv2.putText(resized_image, f"Recording: {label}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        else:
                            #Graphics before recording letting the user know what to sign next
                            resized_image = cv2.putText(resized_image, f"Get ready for: {label}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                        #Displays webcam to user with graphics
                        cv2.imshow("Recording Window", resized_image)
                        #Breaks with 'Esc' key
                        if cv2.waitKey(1) == 27:
                            break
                #Releases the VideoWriter
                result.release()
    #Releases the webcam
    camera.release()
    #Destroys all windows opened by program
    cv2.destroyAllWindows()

#If script is executed from command line
if __name__ == "__main__":
    #Sets up args through parser
    args = setup()

    #Assign values to their global variable
    labels = []
    output_folder_name = args.output
    camera_fps = args.fps
    recording_frames_amount = camera_fps * args.recordtime
    break_time = camera_fps * args.breaktime
    video_amount = args.amount
    camera_width = args.xres
    camera_height = args.yres


    #Add labels argument to list as upper case
    for label in args.Labels:
        labels.append(label.upper())
    
    #Sets up folders
    create_recordings_folder(output_folder_name)
    create_labels(output_folder_name, labels)
    #Calls recording function
    webcam_record(labels, camera_fps, recording_frames_amount, break_time, video_amount, output_folder_name)