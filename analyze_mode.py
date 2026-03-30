# analyze_mode.py
"""
Modified analyze_mode.py to be compatible with the enhanced pose-focused model
and to apply consistent pose normalization based on training parameters.
"""
import os
import json
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# Assuming these are correctly importable from your project structure
import setup_mediapipe # Not directly used in analyze_state but for setting up mp objects
import face_recognition_model as face_model # Used
from frame_processing import process_frame # Used
from multimodal_model import emotions # Used

# Define target states (must match training)
TARGET_STATES = ["Tired", "Bored", "Defensive", "Participating", "Neutral"]

def enhance_pose_features(base_pose_landmarks_flat):
    """
    Create enhanced pose features for a single frame.
    Input base_pose_landmarks_flat contains the 99 base pose features,
    which should be ALREADY NORMALIZED consistently with training.
    """
    if base_pose_landmarks_flat.shape[0] != 99:
        print(f"Error in enhance_pose_features: Expected 99 base features, got {base_pose_landmarks_flat.shape[0]}")
        # This could lead to errors if not handled, e.g., return a zero vector of expected output size
        # or raise an error. For now, we'll assume it proceeds and might error later if shape is wrong.
        # A robust solution would be to return an array of zeros of the expected output shape (105).
        return np.zeros(105) # Example error handling

    landmarks_reshaped = np.array(base_pose_landmarks_flat).reshape(33, 3)
    
    nose = landmarks_reshaped[0]
    left_shoulder = landmarks_reshaped[11]
    right_shoulder = landmarks_reshaped[12]
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    head_tilt = nose - mid_shoulder
    shoulder_slope = right_shoulder - left_shoulder
    
    # Concatenate (already possibly normalized) base features with new derived features
    enhanced_features_vector = np.concatenate([
        base_pose_landmarks_flat, # These are the 99 base features
        head_tilt.flatten(),      # 3 features
        shoulder_slope.flatten()  # 3 features
    ])
    return enhanced_features_vector

def get_state_description(state):
    """Return a description and potential actions for the detected state"""
    descriptions = {
        "Tired": "Signs of fatigue or low energy.",
        "Bored": "Shows disinterest or lack of engagement.",
        "Defensive": "Closed body language, resistance.",
        "Participating": "Actively engaged and focused.",
        "Neutral": "Standard baseline state."
    }
    return descriptions.get(state, "Unknown state")

def analyze_state(args):
    print("\n=== Student State Analysis Mode ===")
    print("Loading models...")
    
    face_model_1, face_model_2 = face_model.load_emotion_models(
        args.face_model_weights_1, args.face_model_weights_2
    )
    
    state_model = None
    norm_params = None
    # <<< START: MODIFIED MODEL AND NORM PARAMS LOADING >>>
    try:
        state_model = load_model(args.state_model_path)
        print(f"State prediction model loaded from {args.state_model_path}")

        base_model_path, _ = os.path.splitext(args.state_model_path)
        norm_params_path = base_model_path + '_norm_params.json'
        
        if os.path.exists(norm_params_path):
            with open(norm_params_path, 'r') as f:
                norm_params = json.load(f)
            print(f"Normalization parameters loaded from {norm_params_path}: {norm_params}")
        else:
            print(f"WARNING: Normalization parameter file not found at {norm_params_path}.")
            print("Pose data will be used raw or as per model's original training if this is an old model.")
            # norm_params remains None, so normalization based on saved params will be skipped.
    except Exception as e:
        print(f"Error loading state model or normalization parameters: {e}")
        return
    # <<< END: MODIFIED MODEL AND NORM PARAMS LOADING >>>
    
    try:
        pose_input_shape_from_model = state_model.inputs[1].shape.as_list()[1]
    except AttributeError: # Keras 3 might use .shape directly
        pose_input_shape_from_model = state_model.inputs[1].shape[1]
        
    print(f"Pose input shape expected by model: {pose_input_shape_from_model}")
    # use_enhanced_features means the model expects more than the raw 99 landmarks (e.g., 105)
    use_enhanced_features_flag = (pose_input_shape_from_model > 99) 
    
    if use_enhanced_features_flag:
        print("Model expects enhanced pose features. Feature enhancement will be applied.")
    else:
        print("Model expects standard (99) pose features. No additional feature enhancement will be applied.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection_obj, \
         mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose_obj:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't read from webcam. Exiting...")
                break
            
            frame = cv2.flip(frame, 1)
            
            (display_frame, current_emotion_idx, current_pose_landmarks_raw, # Raw landmarks from mediapipe
             detected_emotion_label, detected_pose_label, pose_results_mp) = \
                process_frame(frame, face_detection_obj, pose_obj, face_model_1, face_model_2)
            
            if current_emotion_idx is not None and current_pose_landmarks_raw is not None:
                emotion_onehot = tf.keras.utils.to_categorical(
                    [current_emotion_idx], num_classes=len(emotions))
                
                base_pose_flat_raw = np.array(current_pose_landmarks_raw.flatten(), dtype=np.float32) # (99,) raw values
                
                # <<< START: APPLYING NORMALIZATION AND ENHANCEMENT >>>
                processed_base_pose_for_enhancement = base_pose_flat_raw.copy()

                if norm_params and norm_params.get("was_normalized", False):
                    print("Applying training normalization to live base pose data...")
                    pose_min_loaded = norm_params["pose_min"]
                    pose_max_loaded = norm_params["pose_max"]
                    if (pose_max_loaded - pose_min_loaded) != 0:
                        processed_base_pose_for_enhancement = (base_pose_flat_raw - pose_min_loaded) / (pose_max_loaded - pose_min_loaded)
                    else:
                        print("Warning: Loaded pose min and max for scaling are identical. Using 0.5 for normalized values.")
                        processed_base_pose_for_enhancement = np.full_like(base_pose_flat_raw, 0.5)
                else:
                    print("Using raw base pose data (no normalization as per training or params file).")
                
                # Now, decide on final input to model based on whether enhancement is needed
                if use_enhanced_features_flag:
                    # enhance_pose_features takes the (now consistently processed) 99 base features
                    # and returns the 105 feature vector.
                    final_pose_input_to_model = enhance_pose_features(processed_base_pose_for_enhancement)
                else:
                    # If model doesn't use enhanced features, it expects the 99 features
                    final_pose_input_to_model = processed_base_pose_for_enhancement 
                
                final_pose_input_to_model = final_pose_input_to_model.reshape(1, -1) # Reshape for model
                # <<< END: APPLYING NORMALIZATION AND ENHANCEMENT >>>

                # Debugging output
                print(f"Live pose input range (after processing, before model): min={np.min(final_pose_input_to_model):.4f}, max={np.max(final_pose_input_to_model):.4f}")

                state_predictions = state_model.predict([emotion_onehot, final_pose_input_to_model], verbose=0)
                predicted_state_idx = np.argmax(state_predictions[0])
                predicted_state = TARGET_STATES[predicted_state_idx]
                
                cv2.putText(display_frame, f"State: {predicted_state}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
                
                state_description = get_state_description(predicted_state)
                y_pos = 60
                for line in state_description.split('\n'):
                    cv2.putText(display_frame, line, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 20
            else:
                cv2.putText(display_frame, "Cannot predict state - detection incomplete", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if pose_results_mp and pose_results_mp.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame, pose_results_mp.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            cv2.imshow('Student State Analysis', display_frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()