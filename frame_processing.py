import cv2
import face_recognition_model as face_model
from multimodal_model import emotions
import numpy as np

def process_frame(frame, face_detection, pose, face_model_1, face_model_2):
    """Process a single frame to detect face, emotion, and pose"""
    H, W, _ = frame.shape
    display_frame = frame.copy()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize detection variables
    current_emotion_idx = None
    current_pose_landmarks = None
    detected_emotion_label = "None"
    detected_pose_label = "None"
    
    # Face and emotion detection
    face_results_det = face_detection.process(rgb_image)
    if face_results_det.detections:
        detection = face_results_det.detections[0]  # Assume one face
        box = detection.location_data.relative_bounding_box
        x, y, w, h = int(box.xmin*W), int(box.ymin*H), int(box.width*W), int(box.height*H)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x+w, W), min(y+h, H)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            # Prepare for emotion recognition
            face_tensor_resized = face_model.resize_face(face_crop)
            face_batch = face_model.recognition_preprocessing([face_tensor_resized])
            
            # Predict using combined models
            y_1 = face_model_1(face_batch, training=False)
            y_2 = face_model_2(face_batch, training=False)
            combined_pred = y_1 + y_2
            current_emotion_idx = np.argmax(combined_pred, axis=1)[0]
            detected_emotion_label = emotions[current_emotion_idx][0]
            
            # Draw on frame
            emotion_color = emotions[current_emotion_idx][1]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), emotion_color, 2)
            cv2.putText(display_frame, f"Emotion: {detected_emotion_label}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
    
    # Pose detection
    pose_results = pose.process(rgb_image)
    if pose_results.pose_landmarks:
        detected_pose_label = "Detected"
        # Get pose landmarks as numpy array
        current_pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z] 
                                         for lmk in pose_results.pose_landmarks.landmark], 
                                         dtype=np.float32)
        
    return (display_frame, current_emotion_idx, current_pose_landmarks, 
            detected_emotion_label, detected_pose_label, pose_results)
