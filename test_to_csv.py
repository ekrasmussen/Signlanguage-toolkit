import unittest
from model import *
import pandas as pd

class TestToCSVFuctions(unittest.TestCase):
    
    
    def test_confusion_matrix(self):
        # Arrange
        model = YubiModel(15,126,np.array(['A', 'B', 'C', 'D', 'E', 'Idle']), os.path.join('MP_Data'), is_test = True)
        confusion_matrix = [[[45, 1],
                            [1,8]],
                            [[2, 8],
                            [21, 9]],
                            [[84, 6],
                            [35, 4]],
                            [[15, 20],
                            [69, 420]],
                            [[22, 11],
                            [34, 85]],
                            [[78, 1],
                            [1,2]]]
        previous = len(os.listdir(model.logs_path))

        # Act
        model.save_confusion_matrix(confusion_matrix)

        # Assert
        self.assertGreater(len(os.listdir(model.logs_path)), previous)

if __name__ == '__main__':
    unittest.main()