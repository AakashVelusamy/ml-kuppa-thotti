#!/usr/bin/env python3
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os

LOG_PATH = "/home/pi/logs.txt"

def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

print("🎥 Starting Motion Detection (no Flask)...")
log("🎥 Starting Motion Detection (no Flask)...")
# --- Camera setup ---
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(1)
print("✅ Camera started")
log("✅ Camera started")

# --- Motion detection parameters ---
fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=100, detectShadows=True)
MOTION_PIXEL_THRESHOLD = 8000   # Ignore small light changes
MOTION_WAIT = 5                 # Wait 5s after motion to take photo
COOLDOWN = 15                   # Gap between captures (seconds)

motion_detected_time = None
last_capture_time = 0

trigger_file = "/home/pi/motion_trigger.txt"
image_dir = "/home/pi/tmp"
os.makedirs(image_dir, exist_ok=True)

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Background subtraction
        mask = fgbg.apply(gray)
        _, mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.dilate(mask, None, iterations=3)

        motion_score = cv2.countNonZero(mask)
        current_time = time.time()

        # --- Motion detection trigger ---
        if motion_score > MOTION_PIXEL_THRESHOLD:
            if motion_detected_time is None:
                # only start timing if cooldown expired
                if current_time - last_capture_time >= COOLDOWN:
                    motion_detected_time = current_time
                    last_capture_time = current_time  # start cooldown now
                    msg = f"⚠️  Motion detected! (score={motion_score}) — capturing in {MOTION_WAIT}s..."
                    print(msg)
                    log(msg)
                else:
                    msg = "⏳ Cooldown active — skipping motion"
                    print(msg)
                    log(msg)
        else:
            pass  # no motion, continue quietly

        # --- Capture after wait time ---
        if motion_detected_time and (current_time - motion_detected_time >= MOTION_WAIT):
            img_path = f"{image_dir}/captured.jpg"
            cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print(f"📸 Captured → {img_path}")
            log(f"📸 Captured → {img_path}")
            # Signal to classifier
            with open(trigger_file, "w") as f:
                f.write(img_path)

            motion_detected_time = None  # reset after capture

        time.sleep(0.1)

except KeyboardInterrupt:
    print("🛑 Exiting gracefully...")
    log("🛑 Exiting gracefully...")
finally:
    picam2.stop()
    print("📸 Camera stopped.")
    log("📸 Camera stopped.")

