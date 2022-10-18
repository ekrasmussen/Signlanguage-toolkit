
  

# Danish Sign Language Detection
*this project is still in development, certain things are subject to change during the course of development. Please read [our roadmap](#roadmap) for upcoming features and fixes.*
  
  

This repository is a pre-built toolkit aimed for training, recording and testing models for various gestures from the Danish sign language.

# Table of contents
- [Table of contents](#table-of-contents)
- [How it works](#how-it-works)
- [How to use](#how-to-use)
  * [Prerequisites](#prerequisites)
  * [Recorder](#recorder)
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

  

- **Recorder** is used for quickly recording training videos in succession, along with saving them in the correct folders within the parent folder. The program can be customized to use a set interval in between recordings. ***This is a upcoming feature to be pushed to main branch soon***

  

- **Trainer** is used for extracting the data points from the dateset and making them usable as inputs for the model. The trainer is also used for the construction and training of the model.

- **Reader** is available to quickly demo any model made with the trainer, the reader displays the users main webcam, along with the different results and confidence levels.

## Prerequisites
[Python 3.10.7](https://www.python.org/downloads/) Or newer is required. Older versions may work, but has not been tested.

Make sure all package requirements are installed
```
pip install -r requirements.txt
```

## Recorder
*recorder is to be implemented soon, a guide will follow*

## Recorder - GUI

## Trainer
For your training data, create a folder within the parent folder named **"Training_videos"** and create child folders within it named after each of your labels, videos within can be named as you please.

An example of the file structure would be:
```
Repo Folder
└── Training_videos
    ├── cat
    │   ├── video_1.mp4
    │   ├── video_2.mp4
    │   └── ...
    ├── dog
    │   ├── video_1.mp4
    │   ├── video_2.mp4
    │   └── ...
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

## Reader
The reader is launched by using
```
python Webcam.py
```

which automatically loads a model under a specific name at the moment, this is subject to change very soon.

## Reader - GUI

# Roadmap
