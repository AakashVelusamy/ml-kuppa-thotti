import RPi.GPIO as GPIO
import time

TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

print("🔊 HC-SR04 Active — waiting for objects within 25 cm...")

def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    timeout = pulse_start + 0.02
    while GPIO.input(ECHO) == 0 and time.time() < timeout:
        pulse_start = time.time()

    pulse_end = time.time()
    timeout = pulse_end + 0.02
    while GPIO.input(ECHO) == 1 and time.time() < timeout:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = (pulse_duration * 34300) / 2  # cm

    if 2 <= distance <= 400:
        return round(distance, 2)
    else:
        return None

try:
    while True:
        distance = get_distance()
        if distance and distance <= 25:
            print(f"⚠️ Object detected! Distance: {distance} cm")
        time.sleep(0.3)  # Check ~3 times per second

except KeyboardInterrupt:
    print("\n🛑 Exiting...")
finally:
    GPIO.cleanup()

