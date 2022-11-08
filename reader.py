import argparse
from reader_gui import *
from reader_video import *


#Setup arguments
def setup():
    parser = argparse.ArgumentParser(description='Read signs from live video or video recording')
    parser.add_argument('--gui', default=False, action='store_true')
    parser.add_argument('--filepath', type=str, required=False, default='epoch_400_frames_10', help='Path to the file (Default "epoch_400_frames_10")')
    parser.add_argument('--xres', type=int, required=False, default=640, help='Width of camera resolution. (Default 640)')
    parser.add_argument('--yres', type=int, required=False, default=480, help='Height of camera resolution. (Default 480)')
    parser.add_argument('--displayamount', type=int, required=False, default=5, help='The amount of signs displayed when running gui (Default 5)')
    parser.add_argument('--videopath',type=str, required=False, default='Test_video.mov', help='The path to the video that is gonna be read (Default "Test_video.mov")')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = setup()

    #assigns values from arguments
    file_path = args.filepath
    x_res = args.xres
    y_res = args.yres
    display_amount = args.displayamount
    video_path = args.videopath

    #Starts gui if given in args, else start video read
    if args.gui:
        gui = Gui(file_path, x_res, y_res, display_amount)
        gui.setup_gui()
        gui.start()
        gui.root.mainloop()
    else:
        video_reader = VideoReader(file_path, video_path)
        video_reader.start()