import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import time
from collections import deque
from typing import Tuple, Optional, List
import numpy.typing as npt

# ----------------------------
# Settings
# ----------------------------
MODEL_PATH = "yolov8s.pt"
FRAME_WIDTH, FRAME_HEIGHT = 320, 240  # smaller for CPU speed
FRAME_STACK_SIZE = 1
MAX_AGE = 5
MIN_HITS = 2
IOU_THRESH = 0.3
CONF_THRESH = 0.25
MIN_AREA_THRESH = 0.005
SMALL_OBJECT_CLASSES = [39, 41, 46, 67, 76, 84]  # COCO small objects
FOCAL_LENGTH = 1000.0
REAL_OBJECT_SIZE = 0.2

# ----------------------------
# Load YOLO + SORT
# ----------------------------
model = YOLO(MODEL_PATH)
model.fuse()
print("Using CPU for inference")  # CPU only

tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESH)
detection_buffer: deque[npt.NDArray[np.float64]] = deque(maxlen=FRAME_STACK_SIZE)
prev_positions: dict[int, Tuple[int, int, float]] = {}

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# ----------------------------
# Background subtractor
# ----------------------------
fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=15, detectShadows=False)

# ----------------------------
# Utility functions
# ----------------------------
def safe_sort_update(tracker: Sort, dets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if dets.shape[0] == 0 or dets.ndim != 2 or dets.shape[1] != 5:
        return np.empty((0, 5), dtype=np.float64)
    try:
        tracks = tracker.update(dets)
        return tracks if tracks is not None else np.empty((0, 5), dtype=np.float64)
    except:
        return np.empty((0, 5), dtype=np.float64)

def estimate_speed(
    pixel_dist: float, dt: float, focal_length: float, real_object_size: float, pixel_object_size: float
) -> float:
    distance = (real_object_size * focal_length) / max(pixel_object_size, 1e-6)
    real_dist = pixel_dist * (distance / focal_length)
    return real_dist / dt if dt > 0 else 0.0

def preprocess_frame(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(frame, -1, kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

# ----------------------------
# Main loop
# ----------------------------
frame_count = 0
prev_time = time.time()

while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # Ensure frame is np.uint8 for type safety
    frame = np.asarray(frame, dtype=np.uint8)

    frame_count += 1
    h, w, _ = frame.shape
    detections: List[npt.NDArray[np.float64]] = []

    # Preprocess frame to reduce blur
    frame = preprocess_frame(frame)

    # ----------------------------
    # Motion mask
    # ----------------------------
    fgmask = fgbg.apply(frame, learningRate=-1.0)
    fgmask = cv2.medianBlur(fgmask, 5)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_regions: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh > (h * w * MIN_AREA_THRESH):
            motion_regions.append((x, y, x + bw, y + bh))

    # ----------------------------
    # YOLO detection
    # ----------------------------
    if motion_regions:
        # Ensure ROI is np.uint8 as well
        roi = np.asarray(frame, dtype=np.uint8)
        results = model.predict(roi, conf=CONF_THRESH, device="cpu")  # force CPU if CUDA unavailable
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls not in SMALL_OBJECT_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                for mx1, my1, mx2, my2 in motion_regions:
                    if x1 < mx2 and x2 > mx1 and y1 < my2 and y2 > my1:
                        detections.append(np.array([x1, y1, x2, y2, conf], dtype=np.float64))
                        break

    # ----------------------------
    # Add to buffer + SORT
    # ----------------------------
    if detections:
        detection_buffer.append(np.vstack(detections))
    else:
        detection_buffer.append(np.empty((0, 5), dtype=np.float64))

    if detection_buffer:
        combined_dets = np.vstack(detection_buffer) if len(detection_buffer) > 1 else detection_buffer[0]
        if combined_dets.ndim == 2 and combined_dets.shape[1] == 5:
            tracks = safe_sort_update(tracker, combined_dets)
        else:
            tracks = np.empty((0, 5), dtype=np.float64)
    else:
        tracks = np.empty((0, 5), dtype=np.float64)

    # ----------------------------
    # Process tracks and calculate speed
    # ----------------------------
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        if track_id in prev_positions:
            px, py, pt = prev_positions[track_id]
            dt = time.time() - pt
            pixel_dist = np.sqrt((x1 - px) ** 2 + (y1 - py) ** 2)
            pixel_size = max(x2 - x1, y2 - y1)
            speed = estimate_speed(pixel_dist, dt, FOCAL_LENGTH, REAL_OBJECT_SIZE, pixel_size)
        else:
            speed = 0.0
        prev_positions[track_id] = (x1, y1, time.time())

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{track_id} {speed:.2f}m/s",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # ----------------------------
    # Display
    # ----------------------------
    fps = 1 / (time.time() - loop_start + 1e-6)
    cv2.putText(
        frame,
        f"FPS:{int(fps)}",
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    cv2.imshow("Fast Garbage Catcher", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
