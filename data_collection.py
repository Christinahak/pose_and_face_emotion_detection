import numpy as np
import cv2
import mediapipe as mp
import os
import time
import json
from multimodal_model import extract_pose_features

class MultimodalDataCollector:
    """
    Class to collect and save multimodal data for training the combined
    emotion and posture recognition model.
    """
    def __init__(self, output_dir="collected_data"):
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "faces"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "poses"), exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Initialize data structures
        self.samples = []
        self.sample_id = 0
        self.current_emotion = None
        self.current_posture = None
        self.current_context = None
        self.recording = False
        self.sequence_buffer = {
            "faces": [],
            "poses": [],
            "timestamps": []
        }
        
    def start_recording(self, emotion, posture, context="neutral"):
        """Start recording a new sample"""
        self.current_emotion = emotion
        self.current_posture = posture
        self.current_context = context
        self.recording = True
        self.sequence_buffer = {
            "faces": [],
            "poses": [],
            "timestamps": []
        }
        print(f"Started recording sample for Emotion: {emotion}, Posture: {posture}, Context: {context}")
        
    def stop_recording(self):
        """Stop recording and save the sample"""
        if not self.recording:
            print("No active recording to stop")
            return
        
        if len(self.sequence_buffer["faces"]) < 5:
            print("Recording too short, discarding")
            self.recording = False
            return
        
        self.save_sample()
        self.recording = False
        print(f"Sample {self.sample_id} saved.")
        self.sample_id += 1
        
    def process_frame(self, frame):
        """Process a video frame and extract data if recording"""
        H, W, _ = frame.shape
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for face detection
        face_results = self.face_detection.process(rgb_image)
        
        # Process frame for pose detection
        pose_results = self.pose.process(rgb_image)
        
        face_data = None
        pose_data = None
        
        # Extract face data
        if face_results.detections:
            for detection in face_results.detections:
                box = detection.location_data.relative_bounding_box
                x = int(box.xmin * W)
                y = int(box.ymin * H)
                w = int(box.width * W)
                h = int(box.height * H)
                
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, W)
                y2 = min(y + h, H)
                
                face = frame[y1:y2, x1:x2]
                if face.size > 0:  # Ensure face is not empty
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (48, 48))
                    face_data = face
                    break  # Only process the first face
        
        # Extract pose data and features
        if pose_results.pose_landmarks:
            features, pose_array = extract_pose_features(pose_results.pose_landmarks, H, W)
            pose_data = {
                "landmarks": pose_array.tolist(),
                "features": features
            }
        
        # If recording is active and both face and pose data are available, add to buffer
        if self.recording and face_data is not None and pose_data is not None:
            self.sequence_buffer["faces"].append(face_data.tolist())
            self.sequence_buffer["poses"].append(pose_data)
            self.sequence_buffer["timestamps"].append(time.time())
            
        # Return the processed data for visualization
        return face_results, pose_results, face_data, pose_data
            
    def save_sample(self):
        """Save the current sample to disk"""
        sample = {
            "id": self.sample_id,
            "emotion": self.current_emotion,
            "posture": self.current_posture,
            "context": self.current_context,
            "timestamp": time.time(),
            "sequence_length": len(self.sequence_buffer["timestamps"]),
            "timestamps": self.sequence_buffer["timestamps"]
        }
        
        # Save metadata
        with open(os.path.join(self.output_dir, f"sample_{self.sample_id}.json"), 'w') as f:
            json.dump(sample, f)
        
        # Save face data as numpy array
        face_array = np.array(self.sequence_buffer["faces"])
        np.save(os.path.join(self.output_dir, "faces", f"sample_{self.sample_id}.npy"), face_array)
        
        # Save pose data as numpy array (separate landmarks and features)
        landmarks = np.array([p["landmarks"] for p in self.sequence_buffer["poses"]])
        np.save(os.path.join(self.output_dir, "poses", f"sample_{self.sample_id}_landmarks.npy"), landmarks)
        
        # Save feature dictionary separately
        features = [p["features"] for p in self.sequence_buffer["poses"]]
        with open(os.path.join(self.output_dir, "poses", f"sample_{self.sample_id}_features.json"), 'w') as f:
            json.dump(features, f)
        
        # Add to samples list
        self.samples.append(sample)

def prepare_training_data(data_dir="collected_data", sequence_length=10):
    """
    Prepare collected data for training the multimodal model
    
    Args:
        data_dir: Directory containing collected data
        sequence_length: Length of sequences to use for training
        
    Returns:
        faces: Numpy array of face sequences
        poses: Numpy array of pose landmark sequences
        emotions: Numpy array of emotion labels
        postures: Numpy array of posture labels
    """
    # Get all sample metadata
    samples = []
    for filename in os.listdir(data_dir):
        if filename.startswith("sample_") and filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                sample = json.load(f)
                samples.append(sample)
    
    # Sort samples by ID
    samples.sort(key=lambda x: x["id"])
    
    faces = []
    poses = []
    emotions = []
    postures = []
    
    # Process each sample
    for sample in samples:
        sample_id = sample["id"]
        
        # Load face data
        face_file = os.path.join(data_dir, "faces", f"sample_{sample_id}.npy")
        face_seq = np.load(face_file)
        
        # Load pose landmark data
        pose_file = os.path.join(data_dir, "poses", f"sample_{sample_id}_landmarks.npy")
        pose_seq = np.load(pose_file)
        
        # Ensure sequences are of required length
        if len(face_seq) >= sequence_length and len(pose_seq) >= sequence_length:
            # Use the first sequence_length frames
            faces.append(face_seq[:sequence_length])
            poses.append(pose_seq[:sequence_length])
            
            # Add labels
            emotions.append(sample["emotion"])
            postures.append(sample["posture"])
    
    # Convert to numpy arrays
    faces = np.array(faces)
    poses = np.array(poses)
    
    # Reshape faces to add channel dimension
    faces = faces.reshape(faces.shape[0], sequence_length, 48, 48, 1)
    
    # Convert labels to one-hot encoding
    emotion_categories = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    posture_categories = ["defensive", "confident", "anxious", "attentive", "dominant", "neutral"]
    
    emotion_indices = [emotion_categories.index(e.lower()) for e in emotions]
    posture_indices = [posture_categories.index(p.lower()) for p in postures]
    
    emotion_onehot = tf.keras.utils.to_categorical(emotion_indices, num_classes=len(emotion_categories))
    posture_onehot = tf.keras.utils.to_categorical(posture_indices, num_classes=len(posture_categories))
    
    return faces, poses, emotion_onehot, posture_onehot