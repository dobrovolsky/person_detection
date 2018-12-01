import argparse

from multiprocessing import Queue
import cv2
import imutils
from imutils.video import (
    VideoStream,
)
from object_detector import ObjectDetectorRunner

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image")
# ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
ap.add_argument('-v', '--video', help='path to the video file')
args = vars(ap.parse_args())

if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args['video'])

if args.get("video", None) is None:
    vs = VideoStream(src=0).start()

else:
    vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None
frames_q = Queue(maxsize=6)
need_to_detect = Queue(maxsize=100)
t = ObjectDetectorRunner(q=need_to_detect)
t.start()

while True:
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (35, 35), 0)

    if frames_q.full():
        firstFrame = frames_q.get_nowait()
        frames_q.put_nowait(gray)
    else:
        frames_q.put_nowait(gray)
        if firstFrame is None:
            firstFrame = gray
            continue

    # compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    send = False
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            # if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = "Moving"
        send = True

    if send:
        need_to_detect.put(frame_copy)

    # show the frame and record if the user presses a key
    cv2.imshow("Motion color", frame)
    cv2.imshow("Motion gray", gray)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
