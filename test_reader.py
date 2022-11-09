import unittest
from reader_video import *
from load_model import *
import numpy as np


class TestReader(unittest.TestCase):

    #Test if reader_video can predict on a video, and save the sentence to text
    #Requirement is for the corresponding h5 file has a ini file of same name
    def test_video_predict(self):
        
        #arange
        desired_length = 10
        actions = actions = np.array(['A', 'B', 'C', 'D', 'E', 'Idle'])
        file_path = 'epoch_400_frames_10'
        video_path = 'Test_video.mov'

        video_reader = VideoReader(file_path, video_path)

        #act
        dt_string = video_reader.start()
        with open(f'Sentences\Sentence_{dt_string}.txt') as f:
            line = f.readline()
        print(f'line {line}')

        #assert
        self.assertEqual(line, "['A'] \n" )
    
if __name__ == '__main__':
    unittest.main()