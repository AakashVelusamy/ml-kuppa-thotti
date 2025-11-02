import RPi.GPIO as GPIO
import time

TRIG = 24
ECHO = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

print("🔍 Testing Ultrasonic Sensor...")

try:
    while True:
        GPIO.output(TRIG, False)
        time.sleep(0.5)

        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        pulse_start = time.time()
        timeout = pulse_start + 0.04  # 40ms timeout

        while GPIO.input(ECHO) == 0 and time.time() < timeout:
            pulse_start = time.time()

        pulse_end = time.time()
        timeout = pulse_end + 0.04

        while GPIO.input(ECHO) == 1 and time.time() < timeout:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = round(pulse_duration * 17150, 2)

        if 2 < distance < 400:
            print(f"✅ Distance: {distance} cm")
        else:
            print(f"⚠️  Out of range ({distance:.2f} cm) — check alignment or wiring.")

        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Exiting...")
    GPIO.cleanup()

