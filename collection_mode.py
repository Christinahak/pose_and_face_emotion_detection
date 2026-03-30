import face_recognition_model as face_model
import mediapipe as mp
import setup_mediapipe as setup_mp
import cv2
import frame_processing
import time
import json
from multimodal_model import emotions

def collect_data(args):
    """Run data collection mode"""
    print("\n=== Data Collection Mode ===")
    print(f"Saving data to: {args.data_file}")
    print("Press keys to label the CURRENTLY DETECTED emotion/pose:")
    
    # Define keys for target states
    state_keys = {
        ord('t'): "Tired",
        ord('b'): "Bored",
        ord('d'): "Defensive",
        ord('p'): "Participating",
        ord('n'): "Neutral"
    }
    
    for key_ord, state_label in state_keys.items():
        print(f"  '{chr(key_ord)}' = {state_label}")
    print("Press 'q' to quit.")
    print("============================\n")
    
    try:
        # Load models and initialize MediaPipe
        face_model_1, face_model_2 = face_model.load_emotion_models(
            args.face_model_weights_1, args.face_model_weights_2
        )
        face_detection, pose, mp_drawing, mp_drawing_styles = setup_mp.setup_mediapipe()
    except Exception as e:
        print(f"Error initializing models: {e}. Cannot run collection.")
        return
    
    # Setup webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize tracking variables
    last_detected_emotion_idx = None
    last_detected_pose_landmarks = None
    last_save_time = 0
    save_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame.")
            break
        
        H, W, _ = frame.shape
        
        # Process frame to detect face, emotion, and pose
        result = frame_processing.process_frame(frame, face_detection, pose, face_model_1, face_model_2)
        display_frame, current_emotion_idx, current_pose_landmarks, emotion_label, pose_label, pose_results = result
        
        # Draw pose landmarks if available
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_frame, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Update last valid detections
        if current_emotion_idx is not None:
            last_detected_emotion_idx = current_emotion_idx
        if current_pose_landmarks is not None:
            last_detected_pose_landmarks = current_pose_landmarks
        
        # Display status
        cv2.putText(display_frame, f"Detected Emotion: {emotion_label}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(display_frame, f"Detected Pose: {pose_label}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(display_frame, "Press state key (t,b,d,p,n) to SAVE current detection", 
                   (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Saved: {save_count}", 
                   (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.imshow("Data Collection", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('q'):
            break
        elif key in state_keys:
            # Save data with debounce protection
            current_time = time.time()
            if current_time - last_save_time > 0.5:
                target_state = state_keys[key]
                if last_detected_emotion_idx is not None and last_detected_pose_landmarks is not None:
                    if save_collected_instance(args.data_file, current_time, 
                                             last_detected_emotion_idx, 
                                             last_detected_pose_landmarks, 
                                             target_state):
                        print(f"Saved instance: Emotion={emotions[last_detected_emotion_idx][0]}, "
                             f"Pose=Detected, Label='{target_state}'")
                        last_save_time = current_time
                        save_count += 1
                    else:
                        print(f"Failed to save instance for label '{target_state}'")
                else:
                    missing = []
                    if last_detected_emotion_idx is None: missing.append("Emotion")
                    if last_detected_pose_landmarks is None: missing.append("Pose")
                    print(f"Skipping save for '{target_state}': Missing {', '.join(missing)}")
            else:
                print("Saving too fast, skipped.")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Collection finished. Total instances saved: {save_count}")

def save_collected_instance(filename, timestamp, emotion_idx, pose_landmarks, target_state_label):
    """Save a single instance of collected data to a JSON Lines file"""
    data = {
        "timestamp": timestamp,
        "detected_emotion_idx": int(emotion_idx) if emotion_idx is not None else None,
        "pose_landmarks_flat": pose_landmarks.flatten().tolist() if pose_landmarks is not None else None,
        "target_state": target_state_label
    }
    
    try:
        with open(filename, 'a') as f:
            f.write(json.dumps(data) + '\n')
        return True
    except Exception as e:
        print(f"Error saving data instance: {e}")
        return False
