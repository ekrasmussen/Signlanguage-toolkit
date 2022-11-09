import unittest
from reader_video import *
from load_model import *
import numpy as np
from extract_datapoints import *
from model import *
from load_model import *

class TestReadConfig(unittest.TestCase):
    
    def test_train_and_read(self):
        


        #Arrange
        SHAPE = 126
        LENGTH = 2
        ACTIONS = np.array(['A', 'B'])
        VIDEO_PATH = "Training_videos"
        VIDEO_AMOUNT = count_videos(VIDEO_PATH, ACTIONS)
        
        #Act
        extract_data(ACTIONS, VIDEO_AMOUNT, LENGTH, "MP_Data", SHAPE, VIDEO_PATH)
        model = YubiModel(2, SHAPE, ACTIONS, "MP_Data")
        model.train_model(1, VIDEO_AMOUNT, 1)
        configRead = Model(model.timestamp)
        
        #Assert
        self.assertEqual(configRead.shape, SHAPE)
        self.assertEqual(configRead.desired_length, LENGTH)
        self.assertTrue((ACTIONS == configRead.actions).all())



    
if __name__ == '__main__':
    unittest.main()