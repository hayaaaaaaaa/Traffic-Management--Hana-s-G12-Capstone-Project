import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from traffic_lib import EnhancedCarDetector, CarTracker

app = Flask(__name__)

# Initialize AI
car_detector = EnhancedCarDetector(model_type="haar", confidence_threshold=0.5, nms_threshold=0.3)
tracker = CarTracker(max_disappeared=10)

# Global Status
current_status = {
    "lane": "A",
    "ambulance": False,  # <--- NEW FIELD
    "countA": 0,
    "countB": 0
}

def is_ambulance(frame, bbox):
    """
    Check if a detected car is an ambulance based on color (White + Red).
    """
    x, y, w, h = bbox
    # Extract the image of the car only
    car_img = frame[y:y+h, x:x+w]
    
    if car_img.size == 0: return False

    # Convert to HSV color space
    hsv = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)

    # 1. Detect WHITE (Body of ambulance)
    # Sensitivity: Low Saturation, High Value
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = np.count_nonzero(white_mask) / (w * h)

    # 2. Detect RED (Cross/Stripes)
    # Red wraps around 0/180 in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = np.count_nonzero(red_mask) / (w * h)

    # LOGIC: If it's >30% White AND >5% Red -> It's likely the toy ambulance
    if white_ratio > 0.3 and red_ratio > 0.05:
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    global current_status

    if 'image' not in request.files: return "No image", 400
    
    file = request.files['image']
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if frame is None: return "Error", 400

    height, width = frame.shape[:2]
    
    # Detect and Track
    detected_cars = car_detector.detect(frame)
    tracked_cars = tracker.update(detected_cars)

    countA = 0
    countB = 0
    ambulance_found = False
    ambulance_lane = "A"

    for car in tracked_cars:
        cx, cy = car.center
        x, y, w, h = car.bbox
        
        # Determine Lane
        lane_id = "A"
        if cx < (width / 2):
            countA += 1
            lane_id = "A"
        else:
            countB += 1
            lane_id = "B"

        # Check for Ambulance
        if is_ambulance(frame, (x, y, w, h)):
            ambulance_found = True
            ambulance_lane = lane_id
            print(f"[ALERT] Ambulance Detected in Lane {lane_id}!")

    # --- DECISION LOGIC ---
    if ambulance_found:
        # Ambulance overrides everything!
        decision = ambulance_lane
    else:
        # Standard logic
        decision = "A" if countA >= countB else "B"

    current_status = {
        "lane": decision,
        "ambulance": ambulance_found,
        "countA": countA,
        "countB": countB
    }

    print(f"A: {countA} | B: {countB} | Amb: {ambulance_found} -> Winner: {decision}")
    return jsonify(current_status)

@app.route("/decision", methods=["GET"])
def get_decision():
    return jsonify(current_status)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)