#!/usr/bin/env python3
import numpy as np
import cv2
import threading
import time
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
from keras.models import load_model
import os
import sys

LOG_PATH = "/home/pi/logs.txt"
def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

log("\n========== Starting classifier.py ==========")

lcd = CharLCD('PCF8574', 0x27)
lcd.clear()

# --- Custom LCD chars ---
pacman_open = [0b01110,0b11111,0b11000,0b10000,0b11000,0b11111,0b01110,0b00000]
pacman_closed = [0b01110,0b11111,0b11111,0b11111,0b11111,0b11111,0b01110,0b00000]
water_bottle = [0b00100,0b01110,0b01110,0b01110,0b01110,0b01110,0b01110,0b00000]
banana = [0b00100,0b01100,0b01100,0b01100,0b01110,0b00110,0b00010,0b00000]
paperball = [0b00000,0b01110,0b10101,0b01010,0b10101,0b01110,0b00000,0b00000]
pen = [0b00000,0b01100,0b01100,0b00100,0b00100,0b00100,0b00100,0b00000]
dustbin = [0b00000,0b11111,0b11011,0b10101,0b11011,0b10101,0b01110,0b00000]

lcd.create_char(0, pacman_open)
lcd.create_char(1, pacman_closed)
lcd.create_char(2, water_bottle)
lcd.create_char(3, banana)
lcd.create_char(4, paperball)
lcd.create_char(5, pen)
lcd.create_char(6, dustbin)

lcd.clear()
lcd.cursor_pos = (0,0)
lcd.write_string("kuppa thotti " + chr(6))
lcd.cursor_pos = (1,0)
lcd.write_string("activating...")
time.sleep(2)

# --- Load model ---
try:
    model = load_model("/home/pi/model/model.keras")
    CLASSES = ["paper_ball", "banana_peel", "plastic_bottle", "pen"]
    INPUT_SIZE = (128, 128)
    log("Model loaded successfully.")
except Exception as e:
    log(f"Error loading model: {e}")
    lcd.clear()
    lcd.write_string("model fail")
    time.sleep(3)
    sys.exit(1)

lcd.clear()
lcd.cursor_pos = (0,0)
lcd.write_string("kuppa thotti " + chr(6))
lcd.cursor_pos = (1,0)
lcd.write_string("ready!")

# --- Pacman animation ---
animation_running = False
animation_status_text = ""

def pacman_loop():
    items = [2,3,4,5,6]
    while animation_running:
        for pos in range(16):
            if not animation_running:
                break
            lcd.clear()
            for i,it in enumerate(items):
                fpos = 3 + i*3
                if fpos > pos:
                    lcd.cursor_pos = (0, fpos)
                    lcd.write_string(chr(it))
            lcd.cursor_pos = (0,pos)
            lcd.write_string(chr(0 if pos%2==0 else 1))
            lcd.cursor_pos = (1,0)
            lcd.write_string(animation_status_text[:16].ljust(16))
            time.sleep(0.3)
        time.sleep(0.5)

# --- Wait for trigger ---
trigger_file = "/home/pi/motion_trigger.txt"
tmp_dir = "/home/pi/tmp"
os.makedirs(tmp_dir, exist_ok=True)

try:
    while True:
        if os.path.exists(trigger_file):
            with open(trigger_file, "r") as f:
                img_path = f.read().strip()
            os.remove(trigger_file)

            if not os.path.exists(img_path):
                log("Trigger image not found.")
                time.sleep(1)
                continue

            animation_status_text = "predicting..."
            animation_running = True
            t = threading.Thread(target=pacman_loop, daemon=True)
            t.start()

            img = cv2.imread(img_path)
            if img is None:
                log("Failed to read image.")
                animation_running = False
                t.join()
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, INPUT_SIZE)
            img_array = np.expand_dims(img, axis=0)

            preds = model.predict(img_array)
            idx = int(np.argmax(preds))
            pred_class = CLASSES[idx]
            conf = float(np.max(preds)) * 100.0
            log(f"Predicted: {pred_class} ({conf:.2f}%)")

            animation_running = False
            t.join()

            lcd.clear()
            lcd.cursor_pos = (0,0)
            lcd.write_string(pred_class[:16])
            lcd.cursor_pos = (1,0)
            lcd.write_string(f"{conf:.2f}%")
            time.sleep(5)

            lcd.clear()
            lcd.cursor_pos = (0,0)
            lcd.write_string("kuppa thotti " + chr(6))
            lcd.cursor_pos = (1,0)
            lcd.write_string("ready!")

        time.sleep(0.5)

except KeyboardInterrupt:
    log("Stopped manually.")
finally:
    lcd.clear()
    lcd.write_string("goodbye!")
    time.sleep(2)
    lcd.clear()
    GPIO.cleanup()
    log("Exiting gracefully.")

