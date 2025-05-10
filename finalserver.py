from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
from ultralytics import YOLO
import os
import librosa
import sounddevice as sd
import threading
import wavio
import time
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, db
import socket

# Flask app configuration
app_video = Flask(__name__)
app_audio = Flask(__name__)
app_sensor = Flask(__name__)

# Load YOLO models and audio models
model_audio = load_model('C:/Users/think pad/Desktop/five year/senior2/model/Senior_Project/audio_classifier_mfcc_improved.h5')
CATEGORIES = ['Crying', 'Laugh', 'Noise', 'Silence']

model_cry_reason = load_model('C:/Users/think pad/Desktop/five year/senior2/model/Senior_Project/sound_detection_modelCNN.h5')
classes = ['belly_pain', 'burping', 'cold-hot', 'discomfort', "dontKnow", 'hungry', 'lonely', 'scared', 'tired']

yolo_model = YOLO("yolo11n.pt")

# Connection to Firebase
cred = credentials.Certificate("C:/Users/think pad/Desktop/five year/senior2/model/smart-baby-nest-firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-baby-nest-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

hostname = socket.gethostname()  
local_ip = socket.gethostbyname(hostname)  

IP = (f"http://{local_ip}:5000/video_feed")
ref= db.reference('/')
ref.update({
    "Server_IP":IP
})

# Check the status of Manual_control_status from Firebase periodically
manual_control_status = False

def check_manual_control():
    global manual_control_status
    ref = db.reference('/Manual_control_status')
    while True:
        manual_control_status = ref.get()
        print("Manual control status:", manual_control_status)
        if manual_control_status:
            print("Manual control is enabled! Sending signal to Raspberry Pi...")
        time.sleep(10)
        

# Function to async record audio
def record_audio(filename="detected_audio.wav", duration=6, sample_rate=44100, channels=2):
        print("Recording audio...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()
        wavio.write(filename, audio, sample_rate, sampwidth=2)
        return filename

# Function to process audio         
def process_audio(filename):
    try:
        print("Processing audio file...")
        y, sr = librosa.load(filename, sr=None)
        energy = np.sum(y**2) / len(y)  
        rms = librosa.feature.rms(y=y)[0].mean()  
        zcr = librosa.feature.zero_crossing_rate(y)[0].mean()  

        # Validiotion of silence based on energy and rms
        if (energy < 1e-5 and rms < 0.01 and zcr < 0.02) or (zcr < 0.02 and rms < 0.015):
            return "Silence", "No Reason"
        
        prediction = predict_cry(filename)
        if prediction == 'Crying':
            print("Crying detected! Analyzing reason...")
            reason = predict_cry_reason(filename)
            
            ref = db.reference("/")
            ref.update({
                "crying_status": True,
                "Crying_reasons": reason,
                "Manual_control_status": False
            })
        else:
            print("Other sound detected.")
            prediction = "Other Sound"
            reason = "No Reason"
            
            ref = db.reference("/")
            ref.update({
                "crying_status": False,
                "Crying_reasons": reason
            })
        
        return prediction, reason
    except Exception as e:
        print(f"Audio processing error: {e}")
        return "Error", ""
    finally:
        if os.path.exists(filename):
            os.remove(filename)
            print("Audio is processed.")

def predict_cry(file_path):
    mfcc = process_audio_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=(0, -1))
    prediction = model_audio.predict(mfcc)
    return CATEGORIES[np.argmax(prediction)]

def process_audio_mfcc(file_path, duration_seconds=6, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)

def predict_cry_reason(file_path):
    mel_spec = process_audio_mel(file_path)
    mel_spec = np.expand_dims(mel_spec, axis=(0, -1)) / np.max(mel_spec)
    prediction = model_cry_reason.predict(mel_spec)
    return classes[np.argmax(prediction)]

def process_audio_mel(file_path, duration_seconds=6, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    return librosa.power_to_db(mel_spec, ref=np.max)

# Receiving frames from the Raspberry Pi
last_frame = None 

@app_video.route('/upload_frame', methods=['POST'])
def upload_frame():
    global last_frame
    file = request.files['frame']
    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    last_frame = frame 
    child_detected = detect_child(frame)
    
    if not child_detected:
        print("No child detected. Updating crying_status to false.")
        ref = db.reference("/")
        ref.update({
            "crying_status": False,
            "Crying_reasons": "No Reason"
        })
    
    if child_detected:
        print("Child detected! Recording audio...")
        filename = record_audio()
        cry_prediction, cry_reason = process_audio(filename)
        print(f"Cry Detection: {cry_prediction}, Reason: {cry_reason}")
    return jsonify({"child_detected": child_detected})

def detect_child(frame):
    results = yolo_model(frame)
    for box in results[0].boxes:
        cls = box.cls
        conf = box.conf
        if int(cls[0]) == 0 and conf[0] > 0.40:
            return True
    return False

# Video feed for the web app
@app_video.route('/video_feed')
def video_feed():
    def generate():
        global last_frame
        while True:
            if last_frame is not None:
                #Convert the frame into JPEG
                ret, jpeg = cv2.imencode('.jpg', last_frame)
                if ret:
                    # Send the image via HTTP as JPEG
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)  
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Audio analysis to detect cryingand other sounds to send the results to the raspberry pi 
@app_audio.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    print("Starting audio analysis...")
    filename = record_audio()
    cry_prediction, cry_reason = process_audio(filename)
    return jsonify({"Crying": "yes" if cry_prediction == "Crying" or manual_control_status else False, "reason": cry_reason})

# Sensor data to receive temperature and humidity data from the raspberry pi
@app_sensor.route('/upload_sensor_data', methods=['POST'])
def upload_sensor_data():
    data = request.json
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    print("The Temperature and Humidity are received from Raspberry Pi")
    
    # check if all data is present
    if None in (temperature, humidity):
        return jsonify({"error": "Missing data"}), 400

    print(f"Temperature: {temperature} °C, Humidity: {humidity} %")

    # Store data in Firebase Realtime Database
    ref = db.reference('/sensor_data')  # path in the database where the data will be stored
    sensor_data = {
        'temperature': temperature,
        'humidity': humidity,
    }
    ref.update(sensor_data)
    
    return jsonify({"status": "success"}), 200

# Run the Flask app
def run_app(app, port):
    app.run(host='0.0.0.0', port=port)


# Start the Flask app
if __name__ == '__main__':
    video_thread = threading.Thread(target=run_app, args=(app_video, 5000), daemon=True)
    audio_thread = threading.Thread(target=run_app, args=(app_audio, 5001), daemon=True)
    sensor_thread = threading.Thread(target=run_app, args=(app_sensor, 5050), daemon=True)
    
    video_thread.start()
    audio_thread.start()
    sensor_thread.start()

    video_thread.join()
    audio_thread.join()
    sensor_thread.join()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")