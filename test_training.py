import unittest
from detect import *
from model import *
from Training import *
import os


class TestIntegration(unittest.TestCase):

    
    #Checks that the program can extract and train data
    def test_extract_and_train_model(self):


        extract_data(ACTIONS, VIDEO_AMOUNT, DESIRED_LENGTH, DATA_PATH)

        model = YubiModel(DESIRED_LENGTH, SHAPE, ACTIONS, DATA_PATH)
        model.train_model(EPOCHS_AMOUNT, VIDEO_AMOUNT, SEED)


        i = 0
        data_amount = 0
        test_video_amount = 0
        for action in ACTIONS:
            no_sequences = VIDEO_AMOUNT[i]
            test_video_amount += VIDEO_AMOUNT[i]
            for sequence in range (no_sequences):  
                try:
                    data_amount += len(os.listdir(os.path.join(DATA_PATH, action, str(sequence))))

                except:
                    print('except from extract_and_train_model_test in test.training.py')
            i += 1


        #Checks that the correct amount of numpy array are created
        self.assertEqual(data_amount, test_video_amount * DESIRED_LENGTH)

        #Checks that the model has trained
        self.assertGreater(model.n_epochs, 0)

    
if __name__ == '__main__':
    unittest.main()