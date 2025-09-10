import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

class YoloCameraThread(QThread):
    frame_updated = pyqtSignal(np.ndarray, int)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.model = YOLO("yolov8n.pt")  # مدل سبک

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            results = self.model.predict(source=frame, conf=0.5, verbose=False)
            count = 0

            for r in results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if int(cls) == 0:  # کلاس person در COCO
                        count += 1
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            self.frame_updated.emit(frame, count)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
