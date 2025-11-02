from flask import Flask, Response
from picamera2 import Picamera2
import cv2, time, numpy as np

app = Flask(__name__)

# --- Camera setup ---
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(1)
print("✅ Camera started")

# --- Motion detection parameters ---
fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=100, detectShadows=True)
MOTION_PIXEL_THRESHOLD = 8000   # Ignore small light changes
MOTION_WAIT = 5                 # Wait 5s after motion to take photo
COOLDOWN = 15                   # Gap between captures (seconds)

motion_detected_time = None
last_capture_time = 0

def generate_frames():
    global motion_detected_time, last_capture_time

    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Background subtraction
        mask = fgbg.apply(gray)
        # Remove shadows (gray mid tones)
        _, mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
        # Clean small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.dilate(mask, None, iterations=3)

        motion_score = cv2.countNonZero(mask)
        current_time = time.time()

        if motion_score > MOTION_PIXEL_THRESHOLD:
            cv2.putText(frame, f"Motion Detected ({motion_score})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            if motion_detected_time is None:
                motion_detected_time = current_time
                print(f"⚠️ Motion detected (score={motion_score}) — photo in {MOTION_WAIT}s...")
        else:
            cv2.putText(frame, f"No motion ({motion_score})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Capture after 5 seconds if motion was seen and cooldown over
        if motion_detected_time:
            if current_time - motion_detected_time >= MOTION_WAIT:
                if current_time - last_capture_time >= COOLDOWN:
                    cv2.imwrite("./tmp/captured.jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 

                    print("📸 Captured image → captured.jpg")
                    last_capture_time = current_time
                else:
                    print("⏳ Cooldown active — skipping capture")
                motion_detected_time = None

        # Stream output
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("🌐 Open http://<raspberry_pi_ip>:5000/video_feed")
    app.run(host="0.0.0.0", port=5000, threaded=True)
