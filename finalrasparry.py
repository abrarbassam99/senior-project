# To activate the venv: source .venv/bin/activate
# Run the app: python app.py

import time
import threading
import subprocess

import cv2
import requests
import pyaudio
import RPi.GPIO as GPIO
import Adafruit_DHT
import firebase_admin
from firebase_admin import credentials, db

# GPIO Setup
servo_pin = 18
dc_motor_pin = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pin, GPIO.OUT)

servo_pwm = GPIO.PWM(servo_pin, 50)
servo_pwm.start(0)

# Firebase Setup
cred = credentials.Certificate("smart-baby-nest-firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-baby-nest-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Server URLs
server_url_video = "http://192.168.211.235:5000"
server_url_audio = "http://192.168.211.235:5001"
server_url_sensor = "http://192.168.211.235:5050"

# DHT Sensor Setup
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 21  # GPIO pin connected to the sensor

# Global Variables
music_process = None
servo_active = False
servo_thread = None
sensor_thread_active = True  # To control sensor thread activity


# Function to move servo motor
def move_servo(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_pin, True)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(servo_pin, False)
    servo_pwm.ChangeDutyCycle(0)


# Function to rock the crib
def rock_crib():
    while True:
        move_servo(45)
        time.sleep(0.1)
        move_servo(90)
        time.sleep(0.1)
        move_servo(0)
        time.sleep(0.1)


# Function to stream video
def stream_video():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to access camera")
                continue

            _, buffer = cv2.imencode('.jpg', frame)

            # Check server connection before sending data
            try:
                response = requests.post(
                    server_url_video + "/upload_frame",
                    files={"frame": buffer.tobytes()},
                    timeout=20
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("child_detected"):
                        print("Child detected, starting audio analysis...")
                        if result_audio():
                            perform_actions()
                else:
                    print(f"Error: Received invalid status code {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"Error in stream_video: {e}")

            time.sleep(0.1)  # Small delay to slow down frame transmission

    finally:
        cap.release()


# Function to analyze audio and check if the baby is crying
def result_audio():
    try:
        response = requests.post(server_url_audio + "/analyze_audio", timeout=20)
        print("Audio analysis in progress...")

        if response.status_code == 200:
            print(f"Server response: {response.json()}")
            return response.json().get("Crying") == "yes"
        else:
            print(f"Error: Received status code {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Error in result_audio: {e}")
        return False


# Function to execute actions when the baby is crying
def perform_actions():
    global sensor_thread_active

    print("Starting actions for 30 seconds: Music, Servo, and DC Motor.")

    # Stop sensor data collection temporarily during actions
    sensor_thread_active = False

    # Start music
    music_process = subprocess.Popen(["mpg321", "0223.MP3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        start_time = time.time()
        while time.time() - start_time < 30:
            move_servo(45)
            time.sleep(0.1)
            move_servo(90)
            time.sleep(0.1)
            move_servo(0)
            time.sleep(0.1)
            activate_dc_motor()
            time.sleep(1)

        music_process.terminate()
        print("Actions stopped after 30 seconds.")

    finally:
        sensor_thread_active = True  # Resume sensor data collection after actions


# Function to activate DC motor
def activate_dc_motor():
    GPIO.output(dc_motor_pin, GPIO.HIGH)
    time.sleep(1)  # Keep the motor running for 1 second (adjust as needed)
    GPIO.output(dc_motor_pin, GPIO.LOW)


# Temperature and Humidity Sensor reading and sending
def read_and_send_sensor_data():
    while sensor_thread_active:
        try:
            humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
            if humidity is not None and temperature is not None:
                data = {"temperature": temperature, "humidity": humidity}
                try:
                    response = requests.post(server_url_sensor + "/upload_sensor_data", json=data, timeout=5)
                    if response.status_code != 200:
                        print(f"Error: Failed to send sensor data, status code {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Error in sending sensor data: {e}")

            time.sleep(60)  # Sleep for 60 seconds before the next reading

        except Exception as e:
            print(f"Error in read_and_send_sensor_data: {e}")


# Manual control functions for servo, music, and DC motor
def move_servo_continuously():
    global servo_active
    while servo_active:
        move_servo(45)
        time.sleep(0.15)
        move_servo(90)
        time.sleep(0.15)
        move_servo(0)
        time.sleep(0.15)
        activate_dc_motor_manual()


def start_music():
    global music_process
    if music_process is None or music_process.poll() is not None:
        music_process = subprocess.Popen(["mpg321", "0223.MP3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def activate_dc_motor_manual():
    GPIO.output(dc_motor_pin, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(dc_motor_pin, GPIO.LOW)


def manual_control(state):
    global servo_active, servo_thread, music_process

    if state:
        print("Manual mode activated: Starting Music, Servo Motor, and DC Motor")

        # Start music in a separate thread
        music_thread = threading.Thread(target=start_music)
        music_thread.start()

        # Start servo in a separate thread if not already active
        if not servo_active:
            servo_active = True
            servo_thread = threading.Thread(target=move_servo_continuously)
            servo_thread.daemon = True
            servo_thread.start()

        # Activate DC motor in a separate thread
        dc_motor_thread = threading.Thread(target=activate_dc_motor_manual)
        dc_motor_thread.start()

    else:
        print("Manual mode deactivated: Stopping Music, Servo Motor, and DC Motor")

        # Stop music if running
        if music_process is not None and music_process.poll() is None:
            music_process.terminate()
            music_process = None

        # Stop servo if active
        if servo_active:
            servo_active = False  # This will stop the servo thread
            if servo_thread is not None:
                servo_thread.join()  # Wait for the servo thread to finish

        # Stop DC motor
        GPIO.output(dc_motor_pin, GPIO.LOW)


# Firebase listener function for manual control state changes
def listener(event):
    """Callback function triggered when database value changes"""
    manual_state = event.data  # Get new value from database
    if isinstance(manual_state, bool):  # Ensure valid boolean data
        manual_control(manual_state)


# Reference to Firebase node where manual control state is stored
manual_control_ref = db.reference("/Manual_control_status")
manual_control_ref.listen(listener)

# Start threads
sensor_thread = threading.Thread(target=read_and_send_sensor_data)
sensor_thread.start()

video_thread = threading.Thread(target=stream_video)
video_thread.start()

try:
    video_thread.join()
    sensor_thread.join()
finally:
    GPIO.cleanup()