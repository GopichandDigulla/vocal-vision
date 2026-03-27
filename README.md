# Vocal Vision: Real-Time Object Detection with Voice Assistance Using YOLO

## Overview
**Vocal Vision** is a real-time assistive system designed to help visually impaired users identify objects in their surroundings through **audio feedback**.

The project uses:
- **YOLO (Ultralytics)** for real-time object detection
- **OpenCV** for webcam video capture
- **PyTorch** for device support (CPU / CUDA)
- **pyttsx3** for offline text-to-speech output

The system captures frames from a webcam, detects objects, and announces the detected object names through voice.

---
## model

user can download model through this link 
https://drive.google.com/file/d/1RhZeGFNrTTf6QL8sbzidchhoQKhV1aiz/view?usp=sharing

After downloading , place it in the project folder 

## Features
- Real-time object detection using webcam
- Voice output for detected object labels
- Works on **CPU** or **GPU**
- Supports **pretrained YOLO models**
- Can be extended with **custom trained datasets**
- Offline speech output using pyttsx3

---

## Project Workflow
1. Start the program
2. Initialize device (CPU / GPU)
3. Load YOLO model
4. Open webcam
5. Capture video frames continuously
6. Run YOLO object detection on each frame
7. Extract detected object labels
8. Convert labels into speech
9. Display annotated output
10. Stop when `q` is pressed

---

## Folder Structure
```text
project/
│
├── dataset/
│   ├── bag/
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   └── data.yaml
│   ├── book/
│   ├── charger/
│   ├── smarttv/
│   ├── wall/
│   ├── waterbottle/
│   └── data.yaml
│
├── runs/
│   └── detect/
│
├── object.py
├── ob.py
├── requirements.txt
├── yolov8m-world.pt
├── yolov8x-oiv7.pt
└── README.md
```

---

## Technologies Used
- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- pyttsx3
- Roboflow dataset
- Visual Studio Code

---

## Installation

### 1. Create virtual environment
```bash
python -m venv .venv
```

### 2. Activate virtual environment

#### Windows (Command Prompt)
```bash
.venv\Scripts\activate
```

#### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Example requirements.txt
```txt
opencv-python==4.9.0.80
ultralytics==8.2.0
pyttsx3==2.90
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
```

---

## Running the Project
Use:
```bash
python ob.py
```

or:
```bash
python object.py
```

Depending on which file contains your final real-time detection code.

---

## Sample Code
```python
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
```

---

## Dataset Information
The dataset was obtained from **Roboflow** and follows YOLO format.

Typical structure:
```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

Example `data.yaml`:
```yaml
train: dataset/train/images
val: dataset/valid/images
test: dataset/test/images

nc: 6
names: ["bag", "book", "charger", "smart_tv", "wall", "water_bottle"]
```

---

## Training a Custom YOLO Model
```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

The trained weights are usually saved in:
```text
runs/detect/train/weights/best.pt
```

To use the trained model:
```python
model = YOLO("runs/detect/train/weights/best.pt").to(device)
```

---

## Output
The system shows:
- live detection window
- bounding boxes around detected objects
- object labels
- voice announcements of detected objects

---

## Controls
- Press **`q`** to quit the application.

---

## Applications
- Assistive technology for visually impaired users
- Real-time object awareness
- Smart accessibility systems
- AI-based environmental understanding

---

## Future Improvements
- Add distance estimation
- Add obstacle direction guidance
- Support more custom object classes
- Convert into mobile or wearable device application
- Add multilingual voice support

---

## Troubleshooting

### Camera not detected
- Check webcam connection
- Make sure no other app is using the camera
- Try changing:
```python
cv2.VideoCapture(0)
```
or
```python
cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

### No voice output
- Check speaker volume
- Verify `pyttsx3` installation
- Confirm that `sapi5` works on your Windows system

### YOLO model file not found
- Ensure `yolov8m-world.pt` exists in the project folder
- Or give the correct path to the model file

### CUDA not available
- The program will automatically run on CPU if CUDA is unavailable

---

## Authors
- **GOPICHAND DIGULLA**
- **JADI VIVEK SATYA SIDDHARTHA**
- **L. ANUSHA**

**Batch:** A7

---

## Project Title
**Vocal Vision: Real-Time Object Detection with Voice Assistance Using YOLO**
