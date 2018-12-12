import multiprocessing
import os
import time

import cv2
import numpy as np

PATH_YOLO = 'yolo-coco'
CONFIDENCE = 0.5
THRESHOLD = 0.3

labelsPath = os.path.sep.join([PATH_YOLO, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

weightsPath = os.path.sep.join([PATH_YOLO, "yolov3.weights"])
configPath = os.path.sep.join([PATH_YOLO, "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


class ObjectDetector:

    @staticmethod
    def detect(frame):
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (128, 128),
                                     swapRB=True, crop=True)
        net.setInput(blob)
        s = time.time()
        layerOutputs = net.forward(ln)
        print(time.time() - s)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {}".format(LABELS[classIDs[i]],
                                       confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if LABELS[classIDs[i]] == 'person':
                    cv2.imwrite(f'person/{text}.jpg', frame)
                    print(text)


class ObjectDetectorRunner(multiprocessing.Process):
    def __init__(self, q):
        super().__init__()
        self.queue = q

    def run(self):
        while True:
            frame = self.queue.get()
            if isinstance(frame, str):
                break

            ObjectDetector.detect(frame)
