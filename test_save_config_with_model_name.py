import unittest
from model import *
import configparser
import numpy as np

class TestSaveConfigFile(unittest.TestCase):

    def test_save_config(self):
        # Arrange
        actions_list = str(['A', 'B', 'C', 'D', 'E', 'Idle'])
        model = YubiModel(15, 126,np.array(['A', 'B', 'C', 'D', 'E', 'Idle']), os.path.join('MP_Data'), is_test = True)
        config = configparser.ConfigParser()

        # Act
        model.save_as_config_file()

        config.read(f'{model.timestamp}.ini')

        length = config['Model']['Length']
        shape = config['Model']['Shape']
        actions = config['Model']['Actions set']

        # Assert
        self.assertEqual(length, str(15)) 
        self.assertEqual(shape, str(126))
        self.assertTrue(actions, actions_list)
        
if __name__ == '__main__':
    unittest.main()
    