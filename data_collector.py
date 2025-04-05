import os
import cv2
import numpy as np
import time
import json
import mediapipe as mp
from multimodal_model import extract_pose_features

class MultimodalDataCollector:
    def __init__(self, output_dir="collected_data"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "faces"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "poses"), exist_ok=True)
        self.recording = False
        self.sequence_buffer = {"faces": [], "poses": [], "timestamps": []}
        self.sample_id = 0
        self.samples = []
        self.current_combined_state = None
        self.current_context = "neutral"

    def start_recording(self, combined_state, context="neutral"):
        """Start recording a new sample for a combined state"""
        self.current_combined_state = combined_state
        self.current_context = context
        self.recording = True
        self.sequence_buffer = {"faces": [], "poses": [], "timestamps": []} # Reset buffer
        self.start_time = time.time()
        print(f"Started recording sample for Combined State: {combined_state}, Context: {context}")

    def process_frame(self, frame):
        """Process a single frame and add to buffer if recording"""
        H, W, _ = frame.shape
        face_results = None
        pose_results = None
        face_data = None
        pose_data = None

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            face_results = face_detection.process(rgb_image)
            if face_results.detections and self.recording:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = int(bbox.xmin * W), int(bbox.ymin * H), int(bbox.width * W), int(bbox.height * H)
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(x + w, W)
                    y2 = min(y + h, H)
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        self.sequence_buffer["faces"].append(face_gray)

        # Pose detection
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            pose_results = pose.process(rgb_image)
            if pose_results.pose_landmarks and self.recording:
                landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_results.pose_landmarks.landmark])
                features, pose_array = extract_pose_features(pose_results.pose_landmarks, H, W)
                self.sequence_buffer["poses"].append({"landmarks": landmarks.tolist(), "features": features, "gestures": {}}) # Basic gesture placeholder

        if self.recording:
            self.sequence_buffer["timestamps"].append(time.time() - self.start_time)

        return face_results, pose_results, face_data, pose_data

    def stop_recording(self):
        """Stop recording and save the current sample"""
        if self.recording and self.sequence_buffer["poses"]: # Only save if posture data is present
            self.save_sample()
            self.sample_id += 1
            self.recording = False
            print(f"Stopped recording. Sample saved with ID: {self.sample_id - 1}")
        elif self.recording:
            print("No pose data recorded for this sample. Skipping save.")
            self.recording = False

    def save_sample(self):
        """Save the current sample to disk"""
        sample = {
            "id": self.sample_id,
            "combined_state": self.current_combined_state,
            "context": self.current_context,
            "timestamp": time.time(),
            "sequence_length": len(self.sequence_buffer["timestamps"]),
            "timestamps": self.sequence_buffer["timestamps"],
            "gestures": [p["gestures"] for p in self.sequence_buffer["poses"]]
        }

        # Save metadata
        with open(os.path.join(self.output_dir, f"sample_{self.sample_id}.json"), 'w') as f:
            json.dump(sample, f)

        # Save pose data as numpy array (landmarks and features)
        landmarks = np.array([p["landmarks"] for p in self.sequence_buffer["poses"]])
        np.save(os.path.join(self.output_dir, "poses", f"sample_{self.sample_id}_landmarks.npy"), landmarks)

        # Save feature dictionary separately
        features = [p["features"] for p in self.sequence_buffer["poses"]]
        with open(os.path.join(self.output_dir, "poses", f"sample_{self.sample_id}_features.json"), 'w') as f:
            json.dump(features, f)

        # Add to samples list
        self.samples.append(sample)

def prepare_training_data(data_dir, sequence_length):
    all_faces = []
    all_poses = []
    all_emotions = []
    all_postures = []

    # ... (rest of the prepare_training_data function remains the same if you still plan to use it for posture training) ...
    # Note: You might need to adjust this function later if you want to train a model using only posture data.

    return all_faces, all_poses, all_emotions, all_postures