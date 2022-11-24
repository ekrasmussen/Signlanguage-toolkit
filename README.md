
# Danish Sign Language Detection  
  
*this project is still in development, certain things are subject to change during the course of development. Please read [our roadmap](#roadmap) for upcoming features and fixes.*  
  
  
  
This repository is a pre-built toolkit aimed for training, recording and testing models for various gestures from the Danish sign language.  
  
  
  
# Table of contents  
  
- [How it works](#how-it-works)  
- [How to use](#how-to-use)  
  * [Prerequisites](#prerequisites)  
  * [Recorder](#recorder)  
    + [Parameters](#recorder-parameters)  
  * [Recorder - GUI](#recorder---gui)  
  * [Trainer](#trainer)
    + [Parameters](#trainer-parameters) 
    + [First run](#first-run)  
  * [Trainer - GUI](#trainer---gui)  
  * [Reader](#reader)  
  * [Reader - GUI](#reader---gui)  
- [Roadmap](#roadmap)  
  
  
  
  
  
  
  
  
# How it works  
  
  
  
The aim for the project is to combine face, pose and hand position and movement over a set amount of frames. To achieve this, we use [Mediapipe](https://google.github.io/mediapipe/) by Google. More specifically, the project uses the [Holistics](https://google.github.io/mediapipe/solutions/holistic) solution, which is a combination of the face, hands and pose solution.  
  
  
  
The project uses [Tensorflow](https://www.tensorflow.org/) and Keras for construction and training of the model, along with OpenCV for diplaying webcam, showing the data points and results in real time.  
  
  
  
  
  
# How to use  
  
  
  
This project is split up into 3 separate programs. The **Recorder**, **Trainer** and **Reader**. Here is a quick rundown of the 3 programs, and their functions:  
  
  
  
  
  
- **Recorder** is used for quickly recording training videos in succession, along with saving them in the correct folders within the parent folder. The program can be customized to use a set interval in between recordings.  
  
  
  
  
  
- **Trainer** is used for extracting the data points from the dateset and making them usable as inputs for the model. The trainer is also used for the construction and training of the model.  
  
  
  
- **Reader** is available to quickly demo any model made with the trainer, the reader displays the users main webcam, along with the different results and confidence levels.  
  
  
  
## Prerequisites  
  
[Python 3.10.7](https://www.python.org/downloads/) Or newer is required. Older versions may work, but has not been tested.  
  
  
  
Make sure all package requirements are installed  
  
```  
  
pip install -r requirements.txt  
  
```  
  
  
  
## Recorder  
  
The recorder is located inside the Recording directory. The recorder creates a directory (default = "Recordings") as well as the child directories corresponding to the labels which needs to be recorded.  
  
Recorder has one required argument, which is the labels which is going to be recorded. Labels being which actions you are recording  
  
Example:  
```  
python Record.py Label1 Label2 Label3  
```  
In the above example, 3 folders will be created, and you are instructed to record for each label. If you want to record more videos for the same label, simply use it again during a future recording session.  
  
### Recorder Parameters  
Recorder takes certain arguments to customize how the recording session is going to go. It is recommended to use these for more desired results:  
  
- ``--output <Directory name>``: customize the name of the folder which the label directories and videos will reside in, directory will be created if it doesn't exist. Default = "Recordings"  
  
- ``--recordtime <int>``: determine how many seconds each video recording will be. Default = 2  
- ``--breaktime <int>``: determine how many seconds the breaks between recordings are. Default = 3  
- ``--fps <int>`` set this to the closest whole integer for how many frames your camera produces each second. Default = 30  
- ``--amount <int>`` set how many videos will be recorded for each label. Meaning if you have 3 labels and set 2 as the amount, be prepared to record 6 videos. Default = 1  
- ``--xres <int>`` set the x resolution of videos in pixels. Default = 640  
- ``--yres <int>`` set the y resolution of videos in pixels. Default = 480  
  
Example:  
```python Record.py --output MyDirectory --recordtime 5 --breaktime 10 --fps 60 --amount 3 --xres 1920 --yres 1080 Cat Dog```  
In the above example, the videos will be saved in the directory called "MyDirectory", 5 seconds will be recorded fo each video. 10 seconds of break time in between the recordings. The camera is 60 fps, and 3 videos for each label at a resolution of 1920 x 1080. The two labels being "cat" and "dog".  
  
After launching you will have time to prepare. The preparation time amounts to the same amount of time supplied in the ``--breaktime`` parameter  
  
![](https://i.imgur.com/NQqKp4W.png)  
  
During break time, you will also be instructed on what the next label is. After the amount of seconds for break time has passed, recording starts  
  
  
![](https://i.imgur.com/a28xKjg.png)  
  
During recording, the label is still shown.  
  
After all videos are recorded, the program shuts down by itself. Your videos can now be found under your recordings folder.  
  
**Note: If you are using this to gather training data for the trainer. Make sure you verify all videos are good for use (showing as much of the arms and posture as possible, no idle positions or cuts in the middle of the action, and no corrupted videos)**  
## Recorder - GUI  
  
*Planned Feature*  
  
  
  
## Trainer  
  
For your training data, create a folder within the parent folder named **"Training_videos"** and create child folders within it named after each of your labels, videos within can be named as you please.  
  
  
  
An example of the file structure would be:  
  
```  
  
Repo Folder  
  
└── Training_videos  
  
├── cat  
  
│ ├── video_1.mp4  
  
│ ├── video_2.mp4  
  
│ └── ...  
  
├── dog  
  
│ ├── video_1.mp4  
  
│ ├── video_2.mp4  
  
│ └── ...  
  
└── house  
  
├── video_1.mp4  
  
├── video_2.mp4  
  
└── ...  
  
```  
  
### Trainer Parameters  
Trainer takes certain arguments to customize which keypoints are extracted and training customization. It is recommended to use these for more accurate model: 
  
- ``--extract <boolean>``: enables extraction of keypoints. Default = False
  
- ``--gui <boolean>``: launches the gui for training and extraction
- ``--face <boolean>``: tells extractor and trainer to extract/expect face keypoints. Default = False
- ``--pose <boolean>`` tells extractor and trainer to extract/expect pose keypoints. Default = False
- ``--frames <int>`` sets amount of frames that are to be extracted from videos Default = 1
- ``--epochs<int>`` sets amount of epochs used for training Default = 1
- ``--seed<int>`` sets the seed used to split test and training data Default = 1
- ``--path <str>`` sets the path for videos used for training. Default = "Training_videos"
- ``--actionset <str>`` tells extractor and trainer which action set to use. Default = Default  

Note that hand keypoints are always extracted.

Example:  
```python Training.py --extract --face --pose --path 'New_training_vidoes' --actionsset 'Default'```  
In the above example, videos in the directory called "New_training_videos" will have their keypoints for hands, face, and pose extracted in the default amount of desired frames, after which the keypoints will be used for training with default values for amount of frames, epochs and seed for training.
  
### First run  
  
On the first launch of the trainer, and everytime new videos or labels are added into the Training_videos folder, numpy needs to extract the keypoints from each frame and save them. To force the program to extract all keypoints from frames, use  
  
```  
  
python Training.py --extract --face --pose
  
```  
  
This will create a file within the parent directory called "MP_Data", which is the input that the model will be using.  
  
  
  
If all keypoints and frames are extracted, there is no need to extract them again, in which case you can run the script normally by using
  
  
  
```  
  
python Training.py  
  
```  
  
Note: Always train on the same amount of keypoints as extracted. If you want to train with more or less keypoints you need to extract again with the correct parameters.
  
 
  
## Trainer - GUI  
  
The trainer gui is launched by using

 ```  
  
python Training.py --gui  
  
``` 
 
 ![](https://i.imgur.com/qBhbz6K.png)
 Action set: options are 
 - "Default" containing ("A", "B", "C", "D", "E", "IDLE")
 - "Yubi-yay" containing all of Default and ("DROPPER", "HUE", "HVOR", "JUBILÆUM", "SEJR")
**If you are only training make sure that the extracted data matches.**
 
Frame amount: is the amount of frames there is going to be extracted per video, and how many numpy the model trains on. **If you are only training make sure that the extracted data matches.**

Epocs: is how many epocs the model is going to train for.

Seed: is the seed used to split the training data into training and verifying  data.

Extract & train: enables or disables extraction

Key points: choose the keypoint you want to use  **If you are only training make sure that the extracted data matches.**

Select Video Folder: if you have your videos in a different folder you can select that instead.

Select Extract Folder: If you want to extract data to a different folder or already have extracted data in a different folder, you can select that instead.

  
## Reader  
  
The reader is launched by using  
  
```  
  
python reader.py  
  
```  
  

### Reader Parameters  
- ``--gui <boolean>``: launches the gui for reader
- ``--filepath <boolean>``: sets the path to the model(.h5) and config(.ini) Default = "epoch_400_frames_10" **Make sure that your model and config for that model is in the same folder and has the same name**
- ``--xres<int>`` Sets the width of the camera resolution min = 640, Default = 640
- ``--yres<int>`` Sets the height of the camera resolution min = 480, Default = 480
- ``--displayamount<int>`` Sets the amount of actions displayed at a time in the gui min = 1, Default = 5
- ``--videopath<str>`` Sets the path to the video that is going to be read Default = "Test_video.mov"

**If gui is not in parameters then it will try to read the video, if gui is in the parameters it will read from webcam not video**

Example:  
  
``python reader.py --gui --filepath 'new_model' --xres 1920 --yres 1080``
In the above example, the reader will be launched with gui, a model named new_model.h5 and its config file new_model.ini are loaded, and the cameras resolution is set to 1920*1080 (1080p). Note that new_model.h5 and new_model.ini are in the same folder as reader.py.
  
## Reader - GUI  

The reader gui is launched by using 

 ```  
  
python reader.py --gui  
  
``` 
  
![](https://i.imgur.com/8JvvG9s.png)
In the top of the image is the sentence bar it will write your 5 last signs

The signs displayed on the left are the 5 signs the model thinks are most likely to be displayed in percentages. The parameter ``--displayamount<int>`` changes the amount of signs displayed.

Display landmarks: draws the landmarks from hands, pose, and face on the image

Save as text file: Saves the current sentence to a text file

  
  
# Roadmap  
  
- [ ] Beautify Reader GUI  
  
- [x] Create GUI for the trainer, adding ability to customize file paths and training properties  
  
- [x] Create Recorder software  
  
- [x] Add functionality for Reader to also read video files  
  
- [ ] Creation of a bigger dataset  
  
- [ ] Upload of example model for users to test the software without having to train themselves  
  
- [ ] Create a Web Version
