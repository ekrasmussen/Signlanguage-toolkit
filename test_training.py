import unittest
from detect import *
from model import *
from Training import *
import shutil
import os


class TestIntegration(unittest.TestCase):

    
    #Checks that the program can extract and train data
    def test_extract_and_train_model(self):
        # Arrange
        ACTIONS = np.array(['A', 'B', 'C', 'D', 'E', 'Idle', 'DROPPER', 'HVOR', 'HUE', 'JUBILÆUM', 'SEJR'])
        VIDEO_AMOUNT = count_videos("Training_videos", ACTIONS)
        DESIRED_LENGTH = 2
        SEED = 2
        EPOCHS_AMOUNT = 2
        #Remove MP_Data Since it can contain previous videos, meaning the test potentially fails
        shutil.rmtree('MP_Data')

        # Act
        extract_data(ACTIONS, VIDEO_AMOUNT, DESIRED_LENGTH, data_path, 126, "Training_videos")

        model = YubiModel(DESIRED_LENGTH, 126, ACTIONS, data_path)
        model.train_model(EPOCHS_AMOUNT, VIDEO_AMOUNT, SEED)

        i = 0
        data_amount = 0
        test_video_amount = 0
        for action in ACTIONS:
            no_sequences = VIDEO_AMOUNT[i]
            test_video_amount += VIDEO_AMOUNT[i]
            for sequence in range (no_sequences):  
                try:
                    data_amount += len(os.listdir(os.path.join(data_path, action, str(sequence))))
                except:
                    print('Exception from extract_and_train_model_test in test_training.py')
            i += 1

        # Assert
        #Checks that the correct amount of numpy arrays are created
        self.assertEqual(data_amount, test_video_amount * DESIRED_LENGTH)
        #Checks that the model has trained
        self.assertGreater(model.n_epochs, 0)

    
if __name__ == '__main__':
    unittest.main()