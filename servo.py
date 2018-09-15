import RPi.GPIO as GPIO
import time

def initGPIO():
    GPIO.setmode(GPIO.BOARD)
    #servo signal is on pin18
    GPIO.setup(18, GPIO.OUT)
    GPIO.setwarnings(False)
    #set signal frequency to 50hz
    servo = GPIO.PWM(18, 50)
    servo.start(10.3)
    return servo

def actuate(servo):
    time.sleep(0.5)
    servo.ChangeDutyCycle(5.6)
    time.sleep(0.5)

def reset(servo):
    time.sleep(0.5)
    servo.ChangeDutyCycle(10.3)
    time.sleep(0.5)

def cleanup(servo):
    servo.stop()
    GPIO.cleanup()

