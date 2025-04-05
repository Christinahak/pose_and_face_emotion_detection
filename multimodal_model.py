import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import collections
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Concatenate, LSTM, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Define emotions dictionary (same as in your existing code)
emotions = {
    0: ['Angry', (0,0,255), (255,255,255)],
    1: ['Disgust', (0,102,0), (255,255,255)],
    2: ['Fear', (255,255,153), (0,51,51)],
    3: ['Happy', (153,0,153), (255,255,255)],
    4: ['Sad', (255,0,0), (255,255,255)],
    5: ['Surprise', (0,255,0), (255,255,255)],
    6: ['Neutral', (160,160,160), (255,255,255)]
}

# Define postures dictionary
postures = {
    0: ['Defensive', (255,102,102)],
    1: ['Confident', (102,255,102)],
    2: ['Anxious', (255,204,102)],
    3: ['Attentive', (102,178,255)],
    4: ['Dominant', (153,51,255)],
    5: ['Neutral', (192,192,192)]
}

def create_multimodal_emotion_model(face_input_shape=(48, 48, 1), 
                                   pose_input_shape=(33, 3),
                                   num_emotions=7,
                                   num_postures=6,
                                   sequence_length=10):
    """
    Create a multimodal model that combines facial expressions and body posture
    for enhanced emotion and posture recognition.
    """
    # Face branch (CNN) - processes sequence of face images
    face_input = Input(shape=(sequence_length,) + face_input_shape, name='face_input')
    
    face_model = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(face_input)
    face_model = TimeDistributed(BatchNormalization())(face_model)
    face_model = TimeDistributed(MaxPool2D())(face_model)
    face_model = TimeDistributed(Dropout(0.25))(face_model)
    
    face_model = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(face_model)
    face_model = TimeDistributed(BatchNormalization())(face_model)
    face_model = TimeDistributed(MaxPool2D())(face_model)
    face_model = TimeDistributed(Dropout(0.25))(face_model)
    
    face_model = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'))(face_model)
    face_model = TimeDistributed(BatchNormalization())(face_model)
    face_model = TimeDistributed(MaxPool2D())(face_model)
    face_model = TimeDistributed(Dropout(0.25))(face_model)
    
    face_model = TimeDistributed(Flatten())(face_model)
    face_model = TimeDistributed(Dense(256, activation='relu'))(face_model)
    face_model = LSTM(128, return_sequences=False, name='face_lstm')(face_model)
    
    # Pose branch - processes sequence of pose landmarks
    pose_input = Input(shape=(sequence_length,) + pose_input_shape, name='pose_input')

    # Extract meaningful pose features
    pose_model = TimeDistributed(Dense(128, activation='relu'))(pose_input)
    pose_model = TimeDistributed(Dropout(0.3))(pose_model)
    pose_model = TimeDistributed(Dense(64, activation='relu'))(pose_model)
    pose_model = TimeDistributed(Flatten())(pose_model)  # Add this line
    pose_model = LSTM(64, return_sequences=False, name='pose_lstm')(pose_model)
    combined = Concatenate()([face_model, pose_model])
    
    # Joint processing
    combined_features = Dense(256, activation='relu')(combined)
    combined_features = Dropout(0.5)(combined_features)
    combined_features = Dense(128, activation='relu')(combined_features)
    combined_features = Dropout(0.3)(combined_features)
    
    # Two output heads: one for emotion, one for posture
    emotion_output = Dense(num_emotions, activation='softmax', name='emotion_output')(combined_features)
    posture_output = Dense(num_postures, activation='softmax', name='posture_output')(combined_features)
    
    # Create model
    model = Model(
        inputs=[face_input, pose_input],
        outputs=[emotion_output, posture_output]
    )
    
    # Compile model with weighted losses
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={
            'emotion_output': 'categorical_crossentropy',
            'posture_output': 'categorical_crossentropy'
        },
        loss_weights={
            'emotion_output': 1.0,
            'posture_output': 1.0
        },
        metrics=['accuracy']
    )
    
    return model

def angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(p1 - p2)

def chest_region(landmarks, image_height, image_width):
    """Define chest region based on shoulders and hips"""
    left_shoulder = np.array([landmarks[11].x * image_width, landmarks[11].y * image_height])
    right_shoulder = np.array([landmarks[12].x * image_width, landmarks[12].y * image_height])
    left_hip = np.array([landmarks[23].x * image_width, landmarks[23].y * image_height])
    right_hip = np.array([landmarks[24].x * image_width, landmarks[24].y * image_height])
    
    top = min(left_shoulder[1], right_shoulder[1])
    bottom = max(left_shoulder[1], right_shoulder[1]) + (max(left_hip[1], right_hip[1]) - max(left_shoulder[1], right_shoulder[1])) * 0.4
    left = min(left_shoulder[0], left_hip[0])
    right = max(right_shoulder[0], right_hip[0])
    
    return (top, bottom, left, right)

def is_point_in_region(p1, p2, region):
    """Check if both points are in the specified region"""
    top, bottom, left, right = region
    for p in [p1, p2]:
        if not (left <= p[0] <= right and top <= p[1] <= bottom):
            return False
    return True

def extract_pose_features(landmarks, image_height, image_width):
    """
    Extract meaningful features from MediaPipe pose landmarks

    Args:
        landmarks: MediaPipe pose landmarks
        image_height: Height of the image
        image_width: Width of the image

    Returns:
        features: Dictionary of extracted features (currently empty)
        pose_array: Numpy array of landmark coordinates
    """
    features = {}

    # Convert landmarks to numpy array
    pose_array = np.zeros((33, 3))
    if landmarks and landmarks.landmark:  # Add a check if landmarks exist
        for i, landmark in enumerate(landmarks.landmark):
            pose_array[i, 0] = landmark.x
            pose_array[i, 1] = landmark.y
            pose_array[i, 2] = landmark.z

    return features, pose_array

def classify_posture(features):
    """
    Classify posture based on extracted features (currently returns a default).

    Args:
        features: Dictionary of extracted features

    Returns:
        posture_index: Index of the classified posture (defaulting to 5 for 'neutral')
    """
    return 5 # Default to 'neutral'