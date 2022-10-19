import unittest
import numpy as np
from Record import *

class TestsForRecording(unittest.TestCase):

    def test_recorded_videos(self):
        #arrange
        labels = np.array(['Test1', 'Test2'])
        fps = 30
        recordtime = fps * 2
        breaktime = fps * 3
        amount = 2
        output = 'Recordings_test'
        
        #act
        create_recordings_folder(output)
        create_labels(output, labels)
        webcam_record(labels, fps, recordtime, breaktime, amount, output)

        videoAmount = np.zeros(labels.size, dtype=int)
        i = 0
        test_video_amount = 0
        for label in labels:
            videoAmount[i] = len(os.listdir(f'Recordings_test\{label}'))
            test_video_amount += videoAmount[i]
            i += 1

        #assert
        self.assertEqual(test_video_amount, len(labels)*amount)

if __name__ == '__main__':
    unittest.main()