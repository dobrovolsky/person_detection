import argparse
from multiprocessing import Queue

from motion_detector import MotionDetector
from object_detector import ObjectDetectorRunner
from video_reader import VideoStream

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the video file')
args = vars(ap.parse_args())

vs = VideoStream(args['video']).start()

need_to_detect = Queue(maxsize=100)

t = ObjectDetectorRunner(q=need_to_detect)
t.start()

MotionDetector(vs=vs, frame_queue_size=6, detect_queue=need_to_detect).start()
