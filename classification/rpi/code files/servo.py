import RPi.GPIO as GPIO
import time

servo_pin = 17  # GPIO17 (pin 11)

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # 50Hz frequency
pwm.start(0)

def set_angle(angle):
    # Map 0–180° to extended duty cycle 2–13.5
    duty = 2 + (angle / 180) * 10.5
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.7)
    pwm.ChangeDutyCycle(0)

try:
    while True:
        # Clockwise swing (approx -180°)
        set_angle(0)
        time.sleep(1)
        # Counter-clockwise swing (approx +180°)
        set_angle(155)
        time.sleep(1)

except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()

