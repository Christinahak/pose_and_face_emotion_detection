# main.py
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
# Keep necessary TF imports
from tensorflow.keras.models import Sequential, load_model # Keep load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, Rescaling
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import os
import time
import argparse
import json # For saving data in JSON Lines format

# --- Keep VGGNet Definition and Helpers ---
from multimodal_model import emotions # Need emotions dict

# Define your target states based on the keys you want to press
TARGET_STATES = ["Tired", "Bored", "Defensive", "Participating", "Neutral"]
NUM_TARGET_STATES = len(TARGET_STATES)

class VGGNet(Sequential):
    def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
        super().__init__()
        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Flatten())
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(256, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))

        # No compile here, just architecture. Compile when loading/using.
        # Store checkpoint path for loading weights later
        self.checkpoint_path = checkpoint_path

    def compile_model(self, lr=1e-3):
         # Compile method separate from init
         self.compile(optimizer=Adam(learning_rate=lr),
                      loss=categorical_crossentropy,
                      metrics=['accuracy'])


def resize_face(face):
    """Resize face image to 48x48 for model input"""
    # Ensure input is grayscale (H, W)
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    elif len(face.shape) == 3 and face.shape[2] == 1:
         face = face.squeeze(axis=-1) # Remove channel dim if already grayscale

    # Convert to tensor and add channel dim: (H, W) -> (H, W, 1)
    x = tf.expand_dims(tf.convert_to_tensor(face, dtype=tf.float32), axis=2)
    # Resize
    resized = tf.image.resize(x, (48, 48))
    return resized # Return the tensor

def recognition_preprocessing(face_tensors):
    """Preprocess face tensors for the emotion recognition model"""
    # Input should be a list of tensors (each 48, 48, 1)
    # Stack them into a batch
    x = tf.stack(face_tensors)
    return x


# --- Simplified Data Saving Function ---
def save_collected_instance(filename, timestamp, emotion_idx, pose_landmarks, target_state_label):
    """Appends a single instance of collected data to a JSON Lines file."""
    data = {
        "timestamp": timestamp,
        "detected_emotion_idx": int(emotion_idx) if emotion_idx is not None else None, # Ensure native type
        # Flatten landmarks for easier storage/processing later, handle None
        "pose_landmarks_flat": pose_landmarks.flatten().tolist() if pose_landmarks is not None else None,
        "target_state": target_state_label
    }
    try:
        with open(filename, 'a') as f: # Append mode
            f.write(json.dumps(data) + '\n')
        return True
    except Exception as e:
        print(f"Error saving data instance: {e}")
        return False

# --- Function to Create the NEW State Prediction Model ---
def create_state_prediction_model(num_emotions, pose_input_dim, num_states):
    """Creates a model to predict student state from emotion index and pose."""

    # Input for one-hot encoded emotion
    emotion_input = tf.keras.layers.Input(shape=(num_emotions,), name='emotion_input_onehot')

    # Input for flattened pose landmarks
    pose_input = tf.keras.layers.Input(shape=(pose_input_dim,), name='pose_input_flat')

    # Process pose input
    pose_features = tf.keras.layers.Dense(128, activation='relu')(pose_input)
    pose_features = tf.keras.layers.BatchNormalization()(pose_features) # Add normalization
    pose_features = tf.keras.layers.Dropout(0.4)(pose_features)
    pose_features = tf.keras.layers.Dense(64, activation='relu')(pose_features)
    pose_features = tf.keras.layers.BatchNormalization()(pose_features)
    pose_features = tf.keras.layers.Dropout(0.4)(pose_features)


    # Process emotion input (optional, maybe just pass through or small dense)
    emotion_features = tf.keras.layers.Dense(32, activation='relu')(emotion_input) # Small processing layer

    # Combine features
    combined = tf.keras.layers.Concatenate()([emotion_features, pose_features])
    combined = tf.keras.layers.Dense(128, activation='relu')(combined)
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dropout(0.5)(combined)
    combined = tf.keras.layers.Dense(64, activation='relu')(combined)

    # Output layer for state prediction
    output = tf.keras.layers.Dense(num_states, activation='softmax', name='state_output')(combined)

    # Create model
    model = tf.keras.models.Model(inputs=[emotion_input, pose_input], outputs=output, name="StatePredictor")

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adam optimizer
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("State Prediction Model Compiled.")
    model.summary()
    return model

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Multimodal State Analysis")
    parser.add_argument('--mode', choices=['analyze', 'collect', 'train'], required=True, help='Program mode')
    parser.add_argument('--data-file', type=str, default='collected_state_data.jsonl', help='File for storing/reading collected data (JSON Lines format)')
    parser.add_argument('--face-model-weights-1', type=str, default='saved_models/vggnet.h5', help='Path to first VGGNet weights')
    parser.add_argument('--face-model-weights-2', type=str, default='saved_models/vggnet_up.h5', help='Path to second VGGNet weights')
    parser.add_argument('--state-model-path', type=str, default='trained_models/state_predictor.h5', help='Path to save/load the trained state prediction model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training state model') # Increased epochs potentially
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training state model')

    args = parser.parse_args()

    # Ensure output directory for state model exists
    os.makedirs(os.path.dirname(args.state_model_path), exist_ok=True)

    # --- Mode: collect ---
    if args.mode == 'collect':
        print("\n=== Data Collection Mode ===")
        print(f"Saving data to: {args.data_file}")
        print("Press keys to label the CURRENTLY DETECTED emotion/pose:")
        # Define keys for the target states
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

        # --- Initialize Models and MediaPipe ---
        try:
            print("Loading facial emotion models...")
            # Instantiate but don't compile yet
            face_model_1 = VGGNet(input_shape=(48, 48, 1), num_classes=len(emotions), checkpoint_path=args.face_model_weights_1)
            face_model_2 = VGGNet(input_shape=(48, 48, 1), num_classes=len(emotions), checkpoint_path=args.face_model_weights_2)
            face_model_1.load_weights(face_model_1.checkpoint_path)
            face_model_2.load_weights(face_model_2.checkpoint_path)
            # No need to compile VGGNet if only using for prediction
            print("Facial emotion models loaded.")
            use_face_models = True
        except Exception as e:
            print(f"Error loading face models: {e}. Cannot run collection.")
            return

        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

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
            display_frame = frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Reset detections for this frame
            current_emotion_idx = None
            current_pose_landmarks = None
            detected_emotion_label = "None"
            detected_pose_label = "None"

            # --- Face Emotion Detection ---
            face_results_det = face_detection.process(rgb_image)
            if face_results_det.detections:
                detection = face_results_det.detections[0] # Assume one face
                box = detection.location_data.relative_bounding_box
                x, y, w, h = int(box.xmin*W), int(box.ymin*H), int(box.width*W), int(box.height*H)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(x+w, W), min(y+h, H)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    # Prepare for VGGNet using helper functions
                    face_tensor_resized = resize_face(face_crop) # Get the (48,48,1) tensor
                    face_batch = recognition_preprocessing([face_tensor_resized]) # Create batch of 1

                    # Predict using combined models
                    y_1 = face_model_1(face_batch, training=False) # Use call method for inference
                    y_2 = face_model_2(face_batch, training=False)
                    combined_pred = y_1 + y_2
                    current_emotion_idx = np.argmax(combined_pred, axis=1)[0]
                    detected_emotion_label = emotions[current_emotion_idx][0]

                    # Draw on frame
                    emotion_color = emotions[current_emotion_idx][1]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), emotion_color, 2)
                    cv2.putText(display_frame, f"Emotion: {detected_emotion_label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)

            # --- Pose Detection ---
            pose_results = pose.process(rgb_image)
            if pose_results.pose_landmarks:
                detected_pose_label = "Detected"
                mp_drawing.draw_landmarks(
                    display_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                # Store the raw landmarks array
                current_pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z]
                                                  for lmk in pose_results.pose_landmarks.landmark], dtype=np.float32)

            # Store the latest valid detections for saving
            if current_emotion_idx is not None:
                last_detected_emotion_idx = current_emotion_idx
            if current_pose_landmarks is not None:
                last_detected_pose_landmarks = current_pose_landmarks


            # --- Display Status ---
            cv2.putText(display_frame, f"Detected Emotion: {detected_emotion_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(display_frame, f"Detected Pose: {detected_pose_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(display_frame, "Press state key (t,b,d,p,n) to SAVE current detection", (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(display_frame, f"Saved: {save_count}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)


            cv2.imshow("Data Collection", display_frame)

            # --- Keyboard Handling for Saving ---
            key = cv2.waitKey(5) & 0xFF

            if key == ord('q'):
                break
            elif key in state_keys:
                # Save the *last successfully detected* emotion and pose with the user label
                current_time = time.time()
                # Debounce saving - e.g., only save if key pressed > 0.5s after last save
                if current_time - last_save_time > 0.5:
                    target_state = state_keys[key]
                    # Check if we have valid *stored* detections
                    if last_detected_emotion_idx is not None and last_detected_pose_landmarks is not None:
                        if save_collected_instance(args.data_file, current_time, last_detected_emotion_idx, last_detected_pose_landmarks, target_state):
                            print(f"Saved instance: Emotion={emotions[last_detected_emotion_idx][0]}, Pose=Detected, Label='{target_state}'")
                            last_save_time = current_time
                            save_count += 1
                            # Optional: Briefly flash a saved message on screen?
                        else:
                            print(f"Failed to save instance for label '{target_state}'")
                    else:
                        # Inform user why save failed
                        missing = []
                        if last_detected_emotion_idx is None: missing.append("Emotion")
                        if last_detected_pose_landmarks is None: missing.append("Pose")
                        print(f"Skipping save for '{target_state}': Missing {', '.join(missing)}")
                else:
                     # Inform user save was too fast
                     print("Saving too fast, skipped.")


        cap.release()
        cv2.destroyAllWindows()
        print(f"Collection finished. Total instances saved: {save_count}")

    # --- Mode: train ---
    elif args.mode == 'train':
        print("\n=== Training State Prediction Model ===")
        print(f"Loading data from: {args.data_file}")

        # --- Load Data ---
        X_emotion_indices = []
        X_poses_flat = []
        Y_state_labels = []
        pose_dim = 33 * 3 # Expected dimension of flattened pose landmarks

        try:
            with open(args.data_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Validate data before appending
                        if data.get("detected_emotion_idx") is not None and \
                           data.get("pose_landmarks_flat") is not None and \
                           len(data["pose_landmarks_flat"]) == pose_dim and \
                           data.get("target_state") in TARGET_STATES:

                            X_emotion_indices.append(data["detected_emotion_idx"])
                            X_poses_flat.append(data["pose_landmarks_flat"])
                            Y_state_labels.append(data["target_state"])
                        else:
                            print(f"Skipping invalid or incomplete data line: {line.strip()}")
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line: {line.strip()}")
                    except Exception as e:
                         print(f"Error processing line: {line.strip()} - {e}")


            print(f"Loaded {len(Y_state_labels)} valid data instances.")
            if not Y_state_labels:
                 print("No valid data loaded. Cannot train.")
                 return

        except FileNotFoundError:
            print(f"Error: Data file not found at {args.data_file}")
            return

        # --- Preprocess Data ---
        print("Preprocessing data...")
        # Convert emotion indices to one-hot encoding
        num_emotions_available = len(emotions)
        X_emotion_onehot = tf.keras.utils.to_categorical(X_emotion_indices, num_classes=num_emotions_available)

        # Convert flat poses list to NumPy array
        X_poses_np = np.array(X_poses_flat, dtype=np.float32)
        # Optional: Normalize pose data (e.g., StandardScaler) - Fit on training data only!
        # from sklearn.preprocessing import StandardScaler
        # pose_scaler = StandardScaler()
        # X_poses_np = pose_scaler.fit_transform(X_poses_np) # Fit and transform
        # You would need to save this scaler to use it during analysis mode!

        # Convert target state labels to indices, then one-hot
        state_to_index = {label: i for i, label in enumerate(TARGET_STATES)}
        Y_state_indices = [state_to_index[label] for label in Y_state_labels]
        Y_state_onehot = tf.keras.utils.to_categorical(Y_state_indices, num_classes=NUM_TARGET_STATES)

        print(f"Data shapes: Emotions={X_emotion_onehot.shape}, Poses={X_poses_np.shape}, States={Y_state_onehot.shape}")

        # --- Create and Train Model ---
        state_model = create_state_prediction_model(
            num_emotions=num_emotions_available,
            pose_input_dim=pose_dim,
            num_states=NUM_TARGET_STATES
        )

        print("Starting training...")
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                args.state_model_path, # Save directly to the final path
                save_best_only=True,
                monitor='val_accuracy', # Monitor validation accuracy
                mode='max', # Maximize accuracy
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10, # Stop if val_accuracy doesn't improve for 10 epochs
                restore_best_weights=True,
                verbose=1
            ),
             tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', # Reduce LR based on validation loss
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        history = state_model.fit(
            [X_emotion_onehot, X_poses_np], # Inputs as a list
            Y_state_onehot,                # Targets
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.2,          # Use 20% of data for validation
            callbacks=callbacks,
            shuffle=True                   # Shuffle data each epoch
        )

        print(f"Training finished. Best model saved to {args.state_model_path}")

        # Optional: Evaluate on test set if you have one
        # Optional: Plot training history

    # --- Mode: analyze ---
    elif args.mode == 'analyze':
        print("\n=== Analysis Mode ===")
        # --- Load Models ---
        try:
            print("Loading facial emotion models...")
            face_model_1 = VGGNet(input_shape=(48, 48, 1), num_classes=len(emotions), checkpoint_path=args.face_model_weights_1)
            face_model_2 = VGGNet(input_shape=(48, 48, 1), num_classes=len(emotions), checkpoint_path=args.face_model_weights_2)
            face_model_1.load_weights(face_model_1.checkpoint_path)
            face_model_2.load_weights(face_model_2.checkpoint_path)
            print("Facial emotion models loaded.")
        except Exception as e:
            print(f"Error loading face models: {e}. Cannot run analysis.")
            return

        try:
            print(f"Loading trained state prediction model from: {args.state_model_path}")
            state_model = load_model(args.state_model_path)
            print("State prediction model loaded.")
            # Optional: Load pose scaler if you used one during training
            # pose_scaler = joblib.load('pose_scaler.pkl')
        except Exception as e:
            print(f"Error loading state prediction model: {e}. Cannot run analysis.")
            return

        # --- Initialize MediaPipe ---
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            success, frame = cap.read()
            if not success: break

            H, W, _ = frame.shape
            display_frame = frame.copy()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            current_emotion_idx = None
            current_pose_landmarks = None
            predicted_state_label = "Unknown"

            # --- Face Emotion Detection ---
            face_results_det = face_detection.process(rgb_image)
            if face_results_det.detections:
                 detection = face_results_det.detections[0]
                 box = detection.location_data.relative_bounding_box
                 x, y, w, h = int(box.xmin*W), int(box.ymin*H), int(box.width*W), int(box.height*H)
                 x1, y1 = max(0, x), max(0, y)
                 x2, y2 = min(x+w, W), min(y+h, H)
                 face_crop = frame[y1:y2, x1:x2]
                 if face_crop.size > 0:
                     face_tensor_resized = resize_face(face_crop)
                     face_batch = recognition_preprocessing([face_tensor_resized])
                     y_1 = face_model_1(face_batch, training=False)
                     y_2 = face_model_2(face_batch, training=False)
                     current_emotion_idx = np.argmax(y_1 + y_2, axis=1)[0]
                     # Draw detected emotion
                     emotion_label = emotions[current_emotion_idx][0]
                     emotion_color = emotions[current_emotion_idx][1]
                     cv2.rectangle(display_frame, (x1, y1), (x2, y2), emotion_color, 2)
                     cv2.putText(display_frame, f"Emotion: {emotion_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)


            # --- Pose Detection ---
            pose_results = pose.process(rgb_image)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                current_pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_results.pose_landmarks.landmark], dtype=np.float32)

            # --- State Prediction ---
            if current_emotion_idx is not None and current_pose_landmarks is not None:
                # Prepare inputs for state model
                emotion_onehot = tf.keras.utils.to_categorical([current_emotion_idx], num_classes=len(emotions))
                pose_flat = current_pose_landmarks.flatten().reshape(1, -1) # Flatten and add batch dim

                # Optional: Apply pose scaler if used during training
                # pose_flat_scaled = pose_scaler.transform(pose_flat)

                # Predict
                state_predictions = state_model.predict([emotion_onehot, pose_flat], verbose=0) # Use pose_flat_scaled if scaling
                predicted_state_index = np.argmax(state_predictions, axis=1)[0]
                predicted_state_label = TARGET_STATES[predicted_state_index]

            # --- Display Predicted State ---
            cv2.putText(display_frame, f"PREDICTED STATE: {predicted_state_label}", (10, H - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Green color for prediction

            cv2.imshow("State Analysis", display_frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()