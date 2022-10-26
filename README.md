
  

  

# Danish Sign Language Detection

*this project is still in development, certain things are subject to change during the course of development. Please read [our roadmap](#roadmap) for upcoming features and fixes.*

  

This repository is a pre-built toolkit aimed for training, recording and testing models for various gestures from the Danish sign language.

  

# Table of contents

- [How it works](#how-it-works)
- [How to use](#how-to-use)
  * [Prerequisites](#prerequisites)
  * [Recorder](#recorder)
    + [Parameters](#parameters)
  * [Recorder - GUI](#recorder---gui)
  * [Trainer](#trainer)
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

  

  

-  **Recorder** is used for quickly recording training videos in succession, along with saving them in the correct folders within the parent folder. The program can be customized to use a set interval in between recordings.

  

  

-  **Trainer** is used for extracting the data points from the dateset and making them usable as inputs for the model. The trainer is also used for the construction and training of the model.

  

-  **Reader** is available to quickly demo any model made with the trainer, the reader displays the users main webcam, along with the different results and confidence levels.

  

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

### Parameters
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

  

### First run

On the first launch of the trainer, and everytime new videos or labels are added into the Training_videos folder, numpy needs to extract the keypoints from each frame and save them. To force the program to extract all keypoints from frames, use

```

python Training.py --extract

```

This will create a file within the parent directory called "MP_Data", which is the input that the model will be using.

  

If all keypoints and frames are extracted, there is no need to extract them again, in which case you can run the script normally by using

  

```

python Training.py

```

  

**Further customization, like setting epocs, video_length and customizing folders is planned features**

  

## Trainer - GUI

*Planned Feature*

  

## Reader

The reader is launched by using

```

python Webcam.py

```

  

which automatically loads a model under a specific name at the moment, this is subject to change very soon.

  

## Reader - GUI

*Planned Feature*

  

# Roadmap

- [ ] Beautify Reader GUI

- [ ] Create GUI for the trainer, adding ability to customize file paths and training properties

- [x] Create Recorder software

- [ ] Add functionality for Reader to read video files instead of webcam files

- [ ] Creation of a bigger dataset

- [ ] Upload of example model for users to test the software without having to train themselves

- [ ] Create a Web Version
