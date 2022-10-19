import argparse

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
    args = parser.parse_args() 

    return args

def create_recordings_folder(output_folder_name):
    try:
        os.makedirs(output_folder_name)
    except:
        pass

def create_labels(output_folder_name, labels):
    for label in labels:
        try:
            os.makedirs(f"{output_folder_name}/{label}")
        except:
            pass

def webcam_display(labels, fps, recordtime, breaktime, amount, output):
    camera = cv2.VideoCapture(0)
    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if camera.isOpened:
        
        for label in labels:

            for video_number in range(amount):
                
                #Create new Video
                #Uses parameters: path for videos, video codec, frame rate, and image dimensions
                result = cv2.VideoWriter(f"{output}/{label}/{video_number}.mp4", cv2.VideoWriter.fourcc('m','p','4','v'), fps, (camera_width, camera_height))
                
                for frame in range(recordtime + breaktime):
                    ret, image = camera.read()
                    if ret:
                        
                        #Save the image before processing with graphics
                        if frame > breaktime:
                            result.write(image)
                            #graphics
                            image = cv2.putText(image, f"Recording: {label}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        else:
                            #graphics
                            image = cv2.putText(image, f"Get ready for: {label}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


                        cv2.imshow("Recording Window", image)
                        if cv2.waitKey(1) == 27:
                            break
                result.release()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = setup()

    #Assign values to their global variable
    labels = []
    output_folder_name = args.output
    camera_fps = args.fps
    recording_frames_amount = camera_fps * args.recordtime
    break_time = camera_fps * args.breaktime
    video_amount = args.amount

    #Add labels argument to list as upper case
    for label in args.Labels:
        labels.append(label.upper())
    
    create_recordings_folder(output_folder_name)
    create_labels(output_folder_name, labels)
    
    webcam_display(labels, camera_fps, recording_frames_amount, break_time, video_amount, output_folder_name)