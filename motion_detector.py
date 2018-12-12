from queue import Queue

import cv2
import imutils


class MotionDetector:
    def __init__(self, vs, detect_queue, frame_queue_size, display=True, save_to=None, contour_area=1500):
        self.vs = vs
        self.detect_queue = detect_queue

        self._first_frame = None
        self._last_frame = None
        self.frames_q = Queue(maxsize=frame_queue_size)
        self.display = display
        self.save_to = save_to
        self.contour_area = contour_area

    def get_first_frame(self, current_frame):
        if self.frames_q.full():
            self._first_frame = self.frames_q.get_nowait()
            self.frames_q.put_nowait(current_frame)
        else:
            self.frames_q.put_nowait(current_frame)
            if self._first_frame is None:
                self._first_frame = current_frame
                return None
        return self._first_frame

    def read_frame(self):
        frame = self.vs.read()
        if frame is None:
            return
        frame = imutils.resize(frame, width=360)
        self._last_frame = frame
        return self._last_frame

    @staticmethod
    def prepare_frame(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        return frame

    def stop(self):
        self.detect_queue.put('stop')
        self.vs.stop()
        cv2.destroyAllWindows()

    def start(self):
        while True:
            frame = self.read_frame()
            if frame is None:
                continue
            gray = self.prepare_frame(frame)
            first_frame = self.get_first_frame(gray)
            if first_frame is None:
                continue

            frame_copy = frame.copy()

            frame_delta = cv2.absdiff(first_frame, gray)
            thresh = cv2.threshold(frame_delta, 10, 255, cv2.THRESH_BINARY)[1]

            thresh = cv2.dilate(thresh, None, iterations=1)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

            send = False
            for c in cnts:
                if cv2.contourArea(c) < self.contour_area:
                    continue

                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                send = True
                continue

            if send:
                self.detect_queue.put(frame_copy)
                cv2.imwrite(f'person/{id(frame)}.jpg', frame)

            self.save_to and cv2.imwrite(self.save_to, frame)

            key = cv2.waitKey(1) & 0xFF

            if self.display:
                # show the frame and record if the user presses a key
                cv2.imshow("Motion color", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                    self.stop()
                    break
