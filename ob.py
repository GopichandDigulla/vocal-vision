import cv2
from ultralytics import YOLO
import pyttsx3
import torch
import threading
import time

CONFIDENCE_THRESHOLD = 0.25
REPEAT_COOLDOWN = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 170)

def speak(text):
    threading.Thread(
        target=lambda: (engine.say(text), engine.runAndWait()),
        daemon=True
    ).start()

model = YOLO("yolov8m-world.pt").to(device)
# model = YOLO("yolov8x-oiv7.pt").to(device)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not detected")
    exit()

last_spoken = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    if results[0].boxes is not None:
        for box in results[0].boxes:
            label = results[0].names[int(box.cls[0])]
            now = time.time()

            if label not in last_spoken or now - last_spoken[label] > REPEAT_COOLDOWN:
                print("Speaking:", label)
                speak(label)
                last_spoken[label] = now

    cv2.imshow("Live Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()