import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import time

app = Flask(__name__)

# Global Status
status = { "light": "ROAD1", "servo1": 90, "servo2": 90 }
latest_frame = None

# --- SETTINGS ---
# Adjust this if detection is too sensitive or not sensitive enough
# 0 = Black, 255 = White. 
# Anything brighter than this number is considered a "Car"
BRIGHTNESS_THRESHOLD = 60 
MIN_OBJECT_AREA = 1000  # Ignore small specks of dust

@app.route('/')
def index(): return render_template('index.html')

def detect_ambulance_color(img_roi):
    # Simple Red Color Detection
    hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return cv2.countNonZero(mask) > 100 # If > 100 pixels are red

@app.route("/predict", methods=["POST"])
def predict():
    global status, latest_frame
    
    if 'image' not in request.files: return "No image", 400
    
    file = request.files['image']
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if frame is None: return "Error", 400

    height, width = frame.shape[:2]
    cx = width // 2
    cy = height // 2

    # --- THE NEW LOGIC: CONTRAST DETECTION ---
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur it to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Threshold: "If pixel is brighter than X, it's a car"
    # Because your road is black and cars are colorful, this works well.
    _, thresh = cv2.threshold(blurred, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # 4. Find Contours (Shapes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    r1_count, r2_count = 0, 0
    amb_location = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_OBJECT_AREA:
            # We found a car!
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w//2
            center_y = y + h//2
            
            # Check for Ambulance (Red Color inside this box)
            roi = frame[y:y+h, x:x+w]
            is_amb = detect_ambulance_color(roi)
            
            # Draw Box
            color = (0, 0, 255) if is_amb else (0, 255, 0)
            label = "AMB" if is_amb else "Vehicle"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Count Logic
            # Check if it's inside the "Road 1" center strip
            if abs(center_x - cx) < (width * 0.30):
                r1_count += 1
                if is_amb:
                    amb_location = "R1_A" if center_y < cy else "R1_B"
            else:
                r2_count += 1
                if is_amb:
                    amb_location = "R2_A" if center_x < cx else "R2_B"

    # --- TRAFFIC DECISION ---
    srv1, srv2 = 90, 90
    green_light = "ROAD1"

    if amb_location:
        if "R1" in amb_location:
            green_light = "ROAD1"
            srv1 = 180 if amb_location == "R1_A" else 0
        else:
            green_light = "ROAD2"
            srv2 = 180 if amb_location == "R2_A" else 0
    else:
        if r1_count >= r2_count: green_light = "ROAD1"
        else: green_light = "ROAD2"

    status = {"light": green_light, "servo1": srv1, "servo2": srv2}

    # Draw Lines
    cv2.line(frame, (int(cx - width*0.3), 0), (int(cx - width*0.3), height), (255, 0, 0), 2)
    cv2.line(frame, (int(cx + width*0.3), 0), (int(cx + width*0.3), height), (255, 0, 0), 2)
    
    # Dashboard Text
    cv2.putText(frame, f"Signal: {green_light}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"R1: {r1_count} | R2: {r2_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    latest_frame = frame
    return jsonify(status)

@app.route("/decision", methods=["GET"])
def get_decision(): return jsonify(status)

def generate_frames():
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

@app.route('/monitor')
def monitor(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
