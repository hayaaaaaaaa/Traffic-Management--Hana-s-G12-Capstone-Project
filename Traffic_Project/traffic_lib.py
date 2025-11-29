"""
Enhanced Lightweight Car and Lane Detection System for Edge Devices
===================================================================
Fixed version with corrected line unpacking from cv2.HoughLinesP.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import time
import json
import argparse
import sys
import urllib.request
import os
import yaml
from collections import deque
import threading
import queue
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration class for all detection parameters."""
    
    # Lane detection parameters
    roi_ratio: float = 0.6
    min_line_length: int = 50
    smoothing_window: int = 5
    num_lanes: int = 1
    
    # Car detection parameters
    car_model_type: str = "haar"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.3
    
    # Performance parameters
    frame_skip: int = 0
    scale_factor: float = 1.0
    max_frames: Optional[int] = None
    threaded_io: bool = True
    buffer_size: int = 64
    
    # Visualization
    visualize: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'DetectionConfig':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return {
            'roi_ratio': self.roi_ratio,
            'min_line_length': self.min_line_length,
            'smoothing_window': self.smoothing_window,
            'num_lanes': self.num_lanes,
            'car_model_type': self.car_model_type,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'frame_skip': self.frame_skip,
            'scale_factor': self.scale_factor,
            'max_frames': self.max_frames,
            'threaded_io': self.threaded_io,
            'buffer_size': self.buffer_size,
            'visualize': self.visualize
        }


@dataclass
class Lane:
    """Represents a detected lane with its boundaries."""
    id: int
    left_line: np.ndarray
    right_line: np.ndarray
    center_x: float
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point falls within this lane."""
        if len(self.left_line) == 0 or len(self.right_line) == 0:
            return False
        
        try:
            left_sorted = self.left_line[np.argsort(self.left_line[:, 1])]
            right_sorted = self.right_line[np.argsort(self.right_line[:, 1])]
            
            if (y < left_sorted[0, 1] or y > left_sorted[-1, 1] or
                y < right_sorted[0, 1] or y > right_sorted[-1, 1]):
                return False
            
            left_x = np.interp(y, left_sorted[:, 1], left_sorted[:, 0])
            right_x = np.interp(y, right_sorted[:, 1], right_sorted[:, 0])
            
            min_x, max_x = min(left_x, right_x), max(left_x, right_x)
            return min_x <= x <= max_x
            
        except Exception as e:
            logger.debug(f"Error in contains_point: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """Convert Lane to JSON-serializable dictionary."""
        return {
            'id': int(self.id),
            'left_line': self.left_line.tolist() if len(self.left_line) > 0 else [],
            'right_line': self.right_line.tolist() if len(self.right_line) > 0 else [],
            'center_x': float(self.center_x)
        }


@dataclass
class DetectedCar:
    """Represents a detected car with its bounding box."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[float, float]
    lane_id: Optional[int] = None
    track_id: Optional[int] = None
    
    @classmethod
    def from_bbox(cls, x: int, y: int, w: int, h: int, conf: float):
        """Factory method to create a car from bounding box coordinates."""
        center_x = x + w / 2
        center_y = y + h / 2
        return cls(bbox=(x, y, w, h), confidence=conf, center=(center_x, center_y))
    
    def to_dict(self) -> Dict:
        """Convert DetectedCar to JSON-serializable dictionary."""
        return {
            'bbox': {
                'x': int(self.bbox[0]),
                'y': int(self.bbox[1]),
                'width': int(self.bbox[2]),
                'height': int(self.bbox[3])
            },
            'confidence': float(self.confidence),
            'center': {
                'x': float(self.center[0]),
                'y': float(self.center[1])
            },
            'lane_id': int(self.lane_id) if self.lane_id is not None else None,
            'track_id': int(self.track_id) if self.track_id is not None else None
        }


class ThreadedVideoCapture:
    """
    Threaded video capture for improved I/O performance.
    """
    
    def __init__(self, video_path: str, buffer_size: int = 64):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        
        # Video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.fps == 0:
            self.fps = 30
            logger.warning("FPS is 0, using default value of 30")
    
    def start(self):
        """Start the frame reading thread."""
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info("Threaded video capture started")
    
    def _update(self):
        """Thread function to read frames continuously."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Wait if queue is full (backpressure)
            while self.running and self.frame_queue.full():
                time.sleep(0.001)
                
            if self.running:
                self.frame_queue.put((ret, frame))
    
    def read(self):
        """Read a frame from the queue."""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return False, None
    
    def stop(self):
        """Stop the frame reading thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.cap.release()
        logger.info("Threaded video capture stopped")


class ImprovedLaneDetector:
    """
    Enhanced lane detector with temporal smoothing and fixed line unpacking.
    """
    
    def __init__(self, roi_ratio: float = 0.6, min_line_length: int = 50, 
                 smoothing_window: int = 5, num_lanes: int = 1):
        self.roi_ratio = roi_ratio
        self.min_line_length = min_line_length
        self.smoothing_window = smoothing_window
        self.num_lanes = num_lanes
        self.lane_history: deque = deque(maxlen=smoothing_window)
        
        logger.info(f"ImprovedLaneDetector initialized with {smoothing_window}-frame smoothing")

    @lru_cache(maxsize=10)
    def _create_roi_mask(self, height: int, width: int) -> np.ndarray:
        """Cached ROI mask creation."""
        roi_height = int(height * self.roi_ratio)
        
        vertices = np.array([[
            (width * 0.1, height),
            (width * 0.4, roi_height),
            (width * 0.6, roi_height),
            (width * 0.9, height)
        ]], dtype=np.int32)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        return mask

    def _detect_edges_adaptive(self, frame: np.ndarray) -> np.ndarray:
        """Adaptive edge detection with CLAHE."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for adaptive contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive Canny thresholds
        median_intensity = np.median(blur)
        lower = int(max(0, 0.7 * median_intensity))
        upper = int(min(255, 1.3 * median_intensity))
        
        edges = cv2.Canny(blur, lower, upper)
        return edges

    def _detect_lines(self, edges: np.ndarray) -> List[np.ndarray]:
        """Detect lines using Hough Transform."""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=150
        )
        return lines if lines is not None else []

    def _classify_lines_basic(self, lines: List, frame_width: int) -> Tuple[List, List]:
        """Basic line classification (fallback without scikit-learn)."""
        left_lines = []
        right_lines = []
        
        for line in lines:
            # FIX: HoughLinesP returns lines as [[x1, y1, x2, y2]]
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -0.5 and x1 < frame_width / 2:
                left_lines.append(line[0])
            elif slope > 0.5 and x1 > frame_width / 2:
                right_lines.append(line[0])
        
        return [(-1, line) for line in left_lines], [(-1, line) for line in right_lines]

    def _fit_lane_line(self, lines: List, frame_shape: Tuple) -> Optional[np.ndarray]:
        """Fit a polynomial to represent a lane line."""
        if not lines:
            return None
        
        points = []
        for line in lines:
            # Line is already unpacked as [x1, y1, x2, y2] from _classify_lines_basic
            x1, y1, x2, y2 = line
            points.extend([(x1, y1), (x2, y2)])
        
        if len(points) < 2:
            return None
        
        points = np.array(points)
        
        try:
            # Ensure we have enough points for polyfit
            if len(points) < 2:
                return None
                
            coeffs = np.polyfit(points[:, 1], points[:, 0], 1)
        except (np.linalg.LinAlgError, TypeError) as e:
            logger.debug(f"Polyfit error: {e}")
            return None
        
        height = frame_shape[0]
        y1 = height
        y2 = int(height * self.roi_ratio)
        
        x1 = int(coeffs[0] * y1 + coeffs[1])
        x2 = int(coeffs[0] * y2 + coeffs[1])
        
        return np.array([[x1, y1], [x2, y2]])

    def _apply_temporal_smoothing(self, current_lanes: List[Lane]) -> List[Lane]:
        """Apply temporal smoothing to reduce lane jitter."""
        if not current_lanes:
            if self.lane_history:
                return self.lane_history[-1]
            return []
        
        self.lane_history.append(current_lanes)
        
        if len(self.lane_history) < 2:
            return current_lanes
        
        smoothed_lanes = []
        for lane_idx, lane in enumerate(current_lanes):
            historical_left = []
            historical_right = []
            historical_centers = []
            
            for historical_frame in self.lane_history:
                if lane_idx < len(historical_frame):
                    historical_lane = historical_frame[lane_idx]
                    historical_left.append(historical_lane.left_line)
                    historical_right.append(historical_lane.right_line)
                    historical_centers.append(historical_lane.center_x)
            
            # Check if lists are not empty before processing
            if historical_left and historical_right:
                try:
                    # Ensure all arrays have the same shape before averaging
                    left_shapes = [arr.shape for arr in historical_left]
                    right_shapes = [arr.shape for arr in historical_right]
                    
                    if (all(shape == left_shapes[0] for shape in left_shapes) and
                        all(shape == right_shapes[0] for shape in right_shapes)):
                        
                        avg_left = np.mean(historical_left, axis=0).astype(int)
                        avg_right = np.mean(historical_right, axis=0).astype(int)
                        avg_center = np.mean(historical_centers)
                        
                        smoothed_lane = Lane(
                            id=lane.id,
                            left_line=avg_left,
                            right_line=avg_right,
                            center_x=avg_center
                        )
                        smoothed_lanes.append(smoothed_lane)
                    else:
                        # If shapes don't match, use current lane
                        smoothed_lanes.append(lane)
                except Exception as e:
                    logger.debug(f"Error in temporal smoothing: {e}")
                    smoothed_lanes.append(lane)
            else:
                smoothed_lanes.append(lane)
        
        return smoothed_lanes if smoothed_lanes else current_lanes

    def detect(self, frame: np.ndarray) -> List[Lane]:
        """Detect lanes with temporal smoothing and robust error handling."""
        try:
            height, width = frame.shape[:2]
            
            # Get cached ROI mask
            roi_mask = self._create_roi_mask(height, width)
            
            # Detect edges
            edges = self._detect_edges_adaptive(frame)
            masked_edges = cv2.bitwise_and(edges, roi_mask)
            
            # Detect lines
            lines = self._detect_lines(masked_edges)
            
            if len(lines) == 0:
                return self._apply_temporal_smoothing([])
            
            # Use basic classification
            left_clusters, right_clusters = self._classify_lines_basic(lines, width)
            
            # Group clusters by proximity to form lanes
            lanes = []
            lane_id = 0
            
            # For each left cluster, find the closest right cluster
            for left_id, left_lines in self._group_by_cluster(left_clusters):
                left_lane_line = self._fit_lane_line(left_lines, frame.shape)
                if left_lane_line is None:
                    continue
                
                # Find matching right cluster
                best_right_lines = None
                min_distance = float('inf')
                
                for right_id, right_lines in self._group_by_cluster(right_clusters):
                    right_lane_line = self._fit_lane_line(right_lines, frame.shape)
                    if right_lane_line is None:
                        continue
                    
                    # Use explicit array indexing
                    left_x = left_lane_line[0, 0]  # First point, x coordinate
                    right_x = right_lane_line[0, 0]  # First point, x coordinate
                    distance = abs(left_x - right_x)
                    
                    if distance < min_distance and 100 < distance < width * 0.8:
                        min_distance = distance
                        best_right_lines = right_lines
                
                if best_right_lines is not None:
                    right_lane_line = self._fit_lane_line(best_right_lines, frame.shape)
                    if right_lane_line is not None:
                        center_x = (left_lane_line[0, 0] + right_lane_line[0, 0]) / 2
                        lane = Lane(
                            id=lane_id,
                            left_line=left_lane_line,
                            right_line=right_lane_line,
                            center_x=center_x
                        )
                        lanes.append(lane)
                        lane_id += 1
            
            # Apply temporal smoothing
            smoothed_lanes = self._apply_temporal_smoothing(lanes)
            return smoothed_lanes
            
        except Exception as e:
            logger.error(f"Error in improved lane detection: {e}")
            # Return empty lanes instead of crashing
            return []

    def _group_by_cluster(self, clustered_lines):
        """Group lines by their cluster ID."""
        clusters = {}
        for cluster_id, line in clustered_lines:
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(line)
        return clusters.items()


class EnhancedCarDetector:
    """
    Enhanced car detector with NMS and region-based detection.
    """
    
    def __init__(self, model_type: str = "haar", confidence_threshold: float = 0.5, 
                 nms_threshold: float = 0.3):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.car_cascade = None
        self.use_background_subtraction = False
        self.bg_subtractor = None
        
        cascade_loaded = self._load_or_download_cascade()
        
        if not cascade_loaded:
            logger.warning("Using background subtraction fallback")
            self.use_background_subtraction = True
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True
            )
        
        logger.info(f"EnhancedCarDetector initialized with NMS threshold: {nms_threshold}")

    def _load_or_download_cascade(self) -> bool:
        """Load or download Haar cascade for car detection."""
        cascade_paths = [
            'haarcascade_car.xml',
            'cars.xml',
            os.path.join(os.path.dirname(__file__), 'haarcascade_car.xml'),
            os.path.join(os.path.dirname(__file__), 'cars.xml')
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.car_cascade = cascade
                    return True
        
        # Download if not found
        download_path = 'cars.xml'
        url = "https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml"
        
        try:
            urllib.request.urlretrieve(url, download_path)
            cascade = cv2.CascadeClassifier(download_path)
            if not cascade.empty():
                self.car_cascade = cascade
                return True
        except Exception as e:
            logger.debug(f"Download failed: {e}")
        
        return False

    @lru_cache(maxsize=10)
    def _create_roi_mask(self, height: int, width: int) -> np.ndarray:
        """Cached ROI mask creation for car detection."""
        vertices = np.array([[
            (0, height),
            (0, int(height * 0.3)),
            (width, int(height * 0.3)),
            (width, height)
        ]], dtype=np.int32)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        return mask

    def _apply_nms(self, cars: List[DetectedCar]) -> List[DetectedCar]:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if not cars:
            return []
        
        boxes = []
        confidences = []
        
        for car in cars:
            x, y, w, h = car.bbox
            boxes.append([x, y, x + w, y + h])
            confidences.append(car.confidence)
        
        # Handle empty boxes case
        if not boxes:
            return []
            
        try:
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                confidences, 
                self.confidence_threshold, 
                self.nms_threshold
            )
            
            if len(indices) > 0:
                filtered_cars = [cars[i] for i in indices.flatten()]
                logger.debug(f"NMS reduced detections from {len(cars)} to {len(filtered_cars)}")
                return filtered_cars
            else:
                return []
        except Exception as e:
            logger.debug(f"NMS error: {e}, returning original detections")
            return cars

    def _detect_with_background_subtraction(self, frame: np.ndarray) -> List[DetectedCar]:
        """Fallback detection using background subtraction."""
        try:
            fg_mask = self.bg_subtractor.apply(frame)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_cars = []
            height, width = frame.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    if (0.5 < aspect_ratio < 3.0 and 
                        y > height * 0.2 and 
                        x > width * 0.05 and x < width * 0.95):
                        car = DetectedCar.from_bbox(x, y, w, h, conf=0.7)
                        detected_cars.append(car)
            
            return detected_cars
            
        except Exception as e:
            logger.error(f"Error in background subtraction: {e}")
            return []

    def detect(self, frame: np.ndarray) -> List[DetectedCar]:
        """Detect cars with NMS and region-based optimization."""
        try:
            if self.use_background_subtraction:
                cars = self._detect_with_background_subtraction(frame)
                return self._apply_nms(cars)
            
            height, width = frame.shape[:2]
            roi_mask = self._create_roi_mask(height, width)
            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            
            cars_rects = self.car_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(50, 50)
            )
            
            detected_cars = []
            for (x, y, w, h) in cars_rects:
                car = DetectedCar.from_bbox(x, y, w, h, conf=0.8)
                detected_cars.append(car)
            
            filtered_cars = self._apply_nms(detected_cars)
            logger.debug(f"Detected {len(filtered_cars)} cars after NMS")
            return filtered_cars
            
        except Exception as e:
            logger.error(f"Error in enhanced car detection: {e}")
            return []


class CarTracker:
    """
    Centroid-based car tracker for consistent IDs across frames.
    """
    
    def __init__(self, max_disappeared: int = 10):
        self.next_object_id = 0
        self.objects = {}
        self.max_disappeared = max_disappeared
        
    def update(self, detected_cars: List[DetectedCar]) -> List[DetectedCar]:
        """Update tracker with new detections and assign consistent IDs."""
        if len(detected_cars) == 0:
            disappeared_objects = []
            for object_id, (center, bbox, disappeared) in self.objects.items():
                if disappeared + 1 >= self.max_disappeared:
                    disappeared_objects.append(object_id)
                else:
                    self.objects[object_id] = (center, bbox, disappeared + 1)
            
            for object_id in disappeared_objects:
                del self.objects[object_id]
            
            return []
        
        processed_cars = []
        
        if len(self.objects) == 0:
            for car in detected_cars:
                self.objects[self.next_object_id] = (car.center, car.bbox, 0)
                car.track_id = self.next_object_id
                processed_cars.append(car)
                self.next_object_id += 1
            return processed_cars
        
        input_centers = [car.center for car in detected_cars]
        object_ids = list(self.objects.keys())
        object_centers = [self.objects[obj_id][0] for obj_id in object_ids]
        
        used_detections = set()
        
        for i, obj_id in enumerate(object_ids):
            obj_center = object_centers[i]
            min_dist = float('inf')
            best_match = -1
            
            for j, det_center in enumerate(input_centers):
                if j in used_detections:
                    continue
                
                # Explicit array to tuple conversion
                obj_point = np.array(obj_center)
                det_point = np.array(det_center)
                dist = np.linalg.norm(obj_point - det_point)
                
                if dist < min_dist and dist < 50:
                    min_dist = dist
                    best_match = j
            
            if best_match != -1:
                used_detections.add(best_match)
                car = detected_cars[best_match]
                car.track_id = obj_id
                self.objects[obj_id] = (car.center, car.bbox, 0)
                processed_cars.append(car)
        
        for j, car in enumerate(detected_cars):
            if j not in used_detections:
                self.objects[self.next_object_id] = (car.center, car.bbox, 0)
                car.track_id = self.next_object_id
                processed_cars.append(car)
                self.next_object_id += 1
        
        disappeared_objects = []
        for obj_id in object_ids:
            if obj_id not in [car.track_id for car in processed_cars]:
                center, bbox, disappeared = self.objects[obj_id]
                if disappeared + 1 >= self.max_disappeared:
                    disappeared_objects.append(obj_id)
                else:
                    self.objects[obj_id] = (center, bbox, disappeared + 1)
        
        for obj_id in disappeared_objects:
            del self.objects[obj_id]
        
        return processed_cars


class FrameProcessor:
    """
    Enhanced frame processor with tracking and scaling.
    """
    
    def __init__(self, lane_detector: ImprovedLaneDetector, car_detector: EnhancedCarDetector, 
                 scale_factor: float = 1.0):
        self.lane_detector = lane_detector
        self.car_detector = car_detector
        self.car_tracker = CarTracker()
        self.scale_factor = scale_factor
        
        logger.info(f"FrameProcessor initialized with scale factor: {scale_factor}")
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame for faster processing."""
        if self.scale_factor == 1.0:
            return frame
        
        new_width = int(frame.shape[1] * self.scale_factor)
        new_height = int(frame.shape[0] * self.scale_factor)
        return cv2.resize(frame, (new_width, new_height))
    
    def _resize_detections_to_original(self, lanes: List[Lane], cars: List[DetectedCar], 
                                     original_shape: Tuple) -> Tuple[List[Lane], List[DetectedCar]]:
        """Resize detection coordinates back to original frame size."""
        if self.scale_factor == 1.0:
            return lanes, cars
        
        scale_inv = 1.0 / self.scale_factor
        
        resized_lanes = []
        for lane in lanes:
            # Explicit array operations with proper type handling
            left_line = (lane.left_line.astype(float) * scale_inv).astype(int)
            right_line = (lane.right_line.astype(float) * scale_inv).astype(int)
            center_x = lane.center_x * scale_inv
            
            resized_lane = Lane(
                id=lane.id,
                left_line=left_line,
                right_line=right_line,
                center_x=center_x
            )
            resized_lanes.append(resized_lane)
        
        resized_cars = []
        for car in cars:
            x, y, w, h = car.bbox
            resized_bbox = (
                int(x * scale_inv), int(y * scale_inv),
                int(w * scale_inv), int(h * scale_inv)
            )
            center_x, center_y = car.center
            resized_center = (center_x * scale_inv, center_y * scale_inv)
            
            resized_car = DetectedCar(
                bbox=resized_bbox,
                confidence=car.confidence,
                center=resized_center,
                lane_id=car.lane_id,
                track_id=car.track_id
            )
            resized_cars.append(resized_car)
        
        return resized_lanes, resized_cars
    
    def _associate_cars_with_lanes(self, cars: List[DetectedCar], lanes: List[Lane]) -> Dict[int, List[DetectedCar]]:
        """Associate cars with their corresponding lanes."""
        lane_cars = {lane.id: [] for lane in lanes}
        
        for car in cars:
            car_x, car_y = car.center
            
            for lane in lanes:
                if lane.contains_point(car_x, car_y):
                    car.lane_id = lane.id
                    lane_cars[lane.id].append(car)
                    break
        
        return lane_cars
    
    def process(self, frame: np.ndarray) -> Tuple[List[Lane], Dict[int, int], List[DetectedCar]]:
        """Process frame with scaling, detection, and tracking."""
        original_shape = frame.shape
        
        if self.scale_factor != 1.0:
            frame = self._resize_frame(frame)
        
        try:
            lanes = self.lane_detector.detect(frame)
            cars = self.car_detector.detect(frame)
            tracked_cars = self.car_tracker.update(cars)
            lane_cars = self._associate_cars_with_lanes(tracked_cars, lanes)
            
            if self.scale_factor != 1.0:
                lanes, tracked_cars = self._resize_detections_to_original(lanes, tracked_cars, original_shape)
                lane_cars = self._associate_cars_with_lanes(tracked_cars, lanes)
            
            lane_car_counts = {lane_id: len(cars_list) 
                             for lane_id, cars_list in lane_cars.items()}
            
            logger.debug(f"Processed frame: {len(lanes)} lanes, {len(tracked_cars)} tracked cars")
            
            return lanes, lane_car_counts, tracked_cars
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], {}, []


class EnhancedVideoAnalyzer:
    """
    Enhanced video analyzer with all optimizations.
    """
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        
        self.lane_detector = ImprovedLaneDetector(
            roi_ratio=config.roi_ratio,
            min_line_length=config.min_line_length,
            smoothing_window=config.smoothing_window,
            num_lanes=config.num_lanes
        )
        
        self.car_detector = EnhancedCarDetector(
            model_type=config.car_model_type,
            confidence_threshold=config.confidence_threshold,
            nms_threshold=config.nms_threshold
        )
        
        self.frame_processor = FrameProcessor(
            self.lane_detector,
            self.car_detector,
            scale_factor=config.scale_factor
        )
        
        self.visualize = config.visualize
        
        logger.info("EnhancedVideoAnalyzer initialized with all optimizations")
    
    def _draw_lanes(self, frame: np.ndarray, lanes: List[Lane]) -> np.ndarray:
        """Draw detected lanes on frame."""
        overlay = frame.copy()
        colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0)]
        
        for i, lane in enumerate(lanes):
            color = colors[i % len(colors)]
            
            if len(lane.left_line) >= 2:
                # Explicit tuple conversion for OpenCV
                pt1 = tuple(lane.left_line[0].astype(int))
                pt2 = tuple(lane.left_line[1].astype(int))
                cv2.line(overlay, pt1, pt2, color, 3)
            
            if len(lane.right_line) >= 2:
                pt1 = tuple(lane.right_line[0].astype(int))
                pt2 = tuple(lane.right_line[1].astype(int))
                cv2.line(overlay, pt1, pt2, color, 3)
            
            if len(lane.left_line) >= 2 and len(lane.right_line) >= 2:
                pts = np.array([
                    lane.left_line[0],
                    lane.left_line[1],
                    lane.right_line[1],
                    lane.right_line[0]
                ], dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (*color, 100))
        
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    def _draw_cars(self, frame: np.ndarray, cars: List[DetectedCar]) -> np.ndarray:
        """Draw detected cars with tracking IDs."""
        annotated = frame.copy()
        
        for car in cars:
            x, y, w, h = car.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add label with tracking and lane info
            label_parts = []
            if car.track_id is not None:
                label_parts.append(f"ID:{car.track_id}")
            if car.lane_id is not None:
                label_parts.append(f"Lane:{car.lane_id}")
            
            label = " | ".join(label_parts) if label_parts else "No Info"
            
            cv2.putText(
                annotated,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )
        
        return annotated
    
    def _draw_info(self, frame: np.ndarray, lane_counts: Dict[int, int], fps: float, 
                   frame_num: int, total_frames: int) -> np.ndarray:
        """Draw comprehensive information overlay."""
        annotated = frame.copy()
        
        # Draw semi-transparent background
        overlay = annotated.copy()
        text_height = 100 + len(lane_counts) * 30
        cv2.rectangle(overlay, (10, 10), (350, text_height), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
        
        # Draw FPS and frame info
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f} | Frame: {frame_num}/{total_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw car counts per lane
        y_offset = 70
        for lane_id, count in lane_counts.items():
            text = f"Lane {lane_id}: {count} cars"
            cv2.putText(
                annotated,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y_offset += 30
        
        return annotated
    
    def analyze_video(self, video_path: str, output_json_path: Optional[str] = None) -> Dict:
        """Analyze video with all optimizations."""
        try:
            if self.config.threaded_io:
                video_capture = ThreadedVideoCapture(video_path, self.config.buffer_size)
                video_capture.start()
                cap = video_capture
                fps = video_capture.fps
                width = video_capture.width
                height = video_capture.height
                total_frames = video_capture.total_frames
            else:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video: {video_path}")
                
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps == 0:
                    fps = 30
                    logger.warning("FPS is 0, using default value of 30")
            
            logger.info(f"Video opened: {width}x{height} @ {fps} FPS, {total_frames} frames")
            
            results = {
                'video_info': {
                    'path': video_path,
                    'width': int(width),
                    'height': int(height),
                    'fps': int(fps),
                    'total_frames': int(total_frames)
                },
                'config': self.config.to_dict(),
                'frames': [],
                'summary': {
                    'total_frames_processed': 0,
                    'total_cars_detected': 0,
                    'avg_processing_fps': 0,
                    'total_processing_time': 0
                }
            }
            
            frame_count = 0
            processed_count = 0
            fps_history = []
            start_analysis_time = time.time()
            
            while True:
                frame_start_time = time.time()
                
                # Read frame
                if self.config.threaded_io:
                    ret, frame = cap.read()
                else:
                    ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Apply frame skipping
                if self.config.frame_skip > 0 and frame_count % (self.config.frame_skip + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                lanes, lane_counts, cars = self.frame_processor.process(frame)
                
                # Calculate processing FPS
                elapsed = time.time() - frame_start_time
                current_fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_history.append(current_fps)
                
                # Store frame results
                frame_result = {
                    'frame_number': int(frame_count),
                    'timestamp': round(float(frame_count / fps), 3) if fps > 0 else 0.0,
                    'lanes': [lane.to_dict() for lane in lanes],
                    'cars': [car.to_dict() for car in cars],
                    'lane_car_counts': {int(k): int(v) for k, v in lane_counts.items()},
                    'total_cars_in_frame': int(len(cars)),
                    'processing_time_ms': round(float(elapsed * 1000), 2)
                }
                
                results['frames'].append(frame_result)
                results['summary']['total_cars_detected'] += len(cars)
                
                # Visualize if enabled
                if self.visualize:
                    annotated = self._draw_lanes(frame, lanes)
                    annotated = self._draw_cars(annotated, cars)
                    annotated = self._draw_info(annotated, lane_counts, current_fps, 
                                              frame_count, total_frames)
                    
                    cv2.imshow('Enhanced Lane and Car Detection', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User interrupted analysis")
                        break
                
                frame_count += 1
                processed_count += 1
                
                if self.config.max_frames and processed_count >= self.config.max_frames:
                    break
            
            # Calculate summary statistics
            total_processing_time = time.time() - start_analysis_time
            results['summary']['total_frames_processed'] = int(processed_count)
            results['summary']['avg_processing_fps'] = round(float(np.mean(fps_history)), 2) if fps_history else 0.0
            results['summary']['total_processing_time'] = round(float(total_processing_time), 2)
            
            # Cleanup
            if self.config.threaded_io:
                cap.stop()
            else:
                cap.release()
                
            if self.visualize:
                cv2.destroyAllWindows()
            
            # Save to JSON if output path provided
            if output_json_path:
                self._save_results_to_json(results, output_json_path)
            
            logger.info(f"Analysis complete: {processed_count} frames processed, "
                       f"{results['summary']['total_cars_detected']} total cars detected, "
                       f"avg FPS: {results['summary']['avg_processing_fps']:.1f}, "
                       f"total time: {results['summary']['total_processing_time']:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            raise
    
    def _save_results_to_json(self, results: Dict, output_path: str) -> None:
        """Save analysis results to JSON file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON results: {e}")
            raise


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced Lane and Car Detection for Edge Devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py traffic_video.mp4
  
  # With frame skipping and scaling
  python main.py traffic_video.mp4 --skip 2 --scale 0.5
  
  # With visualization
  python main.py traffic_video.mp4 --visualize
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the input video file'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        default=0,
        help='Number of frames to skip between processing (default: 0)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Scale factor for frame processing (0.1-1.0, default: 1.0)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum number of frames to process'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization during processing'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Load configuration
    if args.config:
        config = DetectionConfig.from_yaml(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
    else:
        config = DetectionConfig()
    
    # Override config with command-line arguments
    config.frame_skip = args.skip
    config.scale_factor = args.scale
    config.max_frames = args.max_frames
    config.visualize = args.visualize
    
    # Validate parameters
    if config.scale_factor <= 0 or config.scale_factor > 1:
        logger.error("Scale factor must be between 0.1 and 1.0")
        sys.exit(1)
    
    # Auto-generate output JSON path
    output_json_path = video_path.stem + '_enhanced_results.json'
    
    logger.info(f"Input video: {args.video_path}")
    logger.info(f"Output JSON: {output_json_path}")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Create and run video analyzer
    video_analyzer = EnhancedVideoAnalyzer(config)
    
    try:
        results = video_analyzer.analyze_video(
            video_path=str(video_path),
            output_json_path=output_json_path
        )
        
        # Print summary
        print("\n" + "="*70)
        print("ENHANCED ANALYSIS SUMMARY")
        print("="*70)
        print(f"Frames processed: {results['summary']['total_frames_processed']}")
        print(f"Total cars detected: {results['summary']['total_cars_detected']}")
        print(f"Average processing FPS: {results['summary']['avg_processing_fps']:.2f}")
        print(f"Total processing time: {results['summary']['total_processing_time']:.2f}s")
        print(f"Results saved to: {output_json_path}")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

        
if __name__ == "__main__":
    main()