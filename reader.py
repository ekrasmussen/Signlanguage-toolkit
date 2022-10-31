import argparse
from reader_gui import *

desired_length = 20
actions = np.array(['A', 'B', 'C', 'D', 'E', 'Idle'])
file_path = '31-10-2022 13-45-35.h5'

def setup():
    parser = argparse.ArgumentParser(description='Read signs from live video or video recording')
    parser.add_argument('--gui', default=False, action='store_true')
    parser.add_argument('--desiredlength', type=int, required=False, default=10, help='How many frames the videos are')
    parser.add_argument('--filepath', type=str, required=False, default='epoch_400_frames_10.h5', help='Path to the h5 file')
    parser.add_argument('--xres', type=int, required=False, default=640, help="Width of camera resolution. (Default 640)")
    parser.add_argument('--yres', type=int, required=False, default=480, help="Height of camera resolution. (Default 480)")
    #actions to be added 
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = setup()

    desired_length = args.desiredlength
    file_path = args.filepath
    x_res = args.xres
    y_res = args.yres

    if args.gui:
        gui = Gui(desired_length, actions, file_path, x_res, y_res)
        gui.setup_gui()
        gui.start()
        gui.root.mainloop()