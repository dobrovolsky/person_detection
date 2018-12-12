from threading import Thread

import cv2


class WebcamVideoStream:
    def __init__(self, src=0, name="VideoStream"):
        # initialize the video stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self._update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def _update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class FileVideoStream:
    def __init__(self, path=0, name="VideoStream"):
        # initialize the video stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(path)

        self.name = name

    def start(self):
        return self

    def read(self):
        return self.stream.read()[1]

    def stop(self):
        self.stream.release()


class VideoStream:
    def __init__(self, path=None, src=0):
        if path is not None:
            self.stream = FileVideoStream(path)
        else:
            self.stream = WebcamVideoStream(src=src)

    def start(self):
        return self.stream

    def read(self):
        return self.stream.read()

    def stop(self):
        self.stream.stop()
