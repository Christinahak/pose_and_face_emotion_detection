import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Rescaling  # Updated import
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import os
import time
import collections
import argparse
from multimodal_model import create_multimodal_emotion_model, extract_pose_features, classify_posture, postures

from data_collector import MultimodalDataCollector, prepare_training_data

emotions = {
    0: ['Angry', (0,0,255), (255,255,255)],
    1: ['Disgust', (0,102,0), (255,255,255)],
    2: ['Fear', (255,255,153), (0,51,51)],
    3: ['Happy', (153,0,153), (255,255,255)],
    4: ['Sad', (255,0,0), (255,255,255)],
    5: ['Surprise', (0,255,0), (255,255,255)],
    6: ['Neutral', (160,160,160), (255,255,255)]
}

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

        self.compile(optimizer=Adam(learning_rate=lr),
                      loss=categorical_crossentropy,
                      metrics=['accuracy'])

        self.checkpoint_path = checkpoint_path

def resize_face(face):
    """Resize face image to 48x48 for model input"""
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, (48, 48))

def recognition_preprocessing(faces):
    """Preprocess face images for the emotion recognition model"""
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x

class MultimodalAnalyzer:
    """
    Class for real-time multimodal emotion and posture analysis
    using facial expressions and body posture
    """
    def __init__(self, model_path=None, sequence_length=10, use_rule_based=True,
                 face_model_weights_1='saved_models/vggnet.h5',
                 face_model_weights_2='saved_models/vggnet_up.h5'):
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing_face = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing_pose = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        # Set up sequence buffers
        self.sequence_length = sequence_length
        self.face_sequence = collections.deque(maxlen=sequence_length)
        self.pose_sequence = collections.deque(maxlen=sequence_length)

        # Load or initialize multimodal model
        self.use_rule_based = use_rule_based  # Whether to use rule-based posture classification

        if model_path and os.path.exists(model_path):
            print(f"Loading multimodal model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("Initializing new multimodal model (not trained)")
            self.model = create_multimodal_emotion_model(
                sequence_length=sequence_length
            )

        # Load your facial emotion recognition models
        self.face_model_1 = VGGNet(input_shape=(48, 48, 1), num_classes=len(emotions), checkpoint_path=face_model_weights_1)
        self.face_model_2 = VGGNet(input_shape=(48, 48, 1), num_classes=len(emotions), checkpoint_path=face_model_weights_2)
        self.use_separate_face_model = False # Initialize to False

        if os.path.exists(face_model_weights_1) and os.path.exists(face_model_weights_2):
            try:
                self.face_model_1.load_weights(self.face_model_1.checkpoint_path)
                self.face_model_2.load_weights(self.face_model_2.checkpoint_path)
                print("Facial emotion recognition models loaded.")
                self.use_separate_face_model = True
            except Exception as e:
                print(f"Error loading facial emotion recognition models: {e}")
        else:
            print("Warning: Facial emotion recognition model weights not found.")

    def process_frame(self, frame):
        H, W, _ = frame.shape
        display_frame = frame.copy()
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process for face detection
        face_results = self.face_detection.process(rgb_image)

        predicted_emotion_index = None
        if face_results.detections and self.use_separate_face_model:
            faces = []
            positions = []
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
                if face.size > 0:
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48))
                    faces.append(face_resized)
                    positions.append((x1, y1, x2, y2))

            if faces:
                x_face = recognition_preprocessing(faces)
                y_1_face = self.face_model_1.predict(x_face, verbose=0)
                y_2_face = self.face_model_2.predict(x_face, verbose=0)
                emotion_pred_face = np.argmax(y_1_face + y_2_face, axis=1)[0]
                predicted_emotion_index = emotion_pred_face

                # Display detected face emotion
                if predicted_emotion_index is not None and positions:
                    cv2.putText(display_frame, f"Face Emotion: {emotions[predicted_emotion_index][0]}",
                                (positions[0][0], positions[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                emotions[predicted_emotion_index][1], 2, cv2.LINE_AA)
                    cv2.rectangle(display_frame, (positions[0][0], positions[0][1]),
                                  (positions[0][2], positions[0][3]), emotions[predicted_emotion_index][1], 2, cv2.LINE_AA)

        # Process for pose detection
        pose_results = self.pose.process(rgb_image)

        # Draw pose landmarks and extract features (as before)
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                display_frame,
                pose_results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

            pose_features, pose_array = extract_pose_features(
                pose_results.pose_landmarks,
                H, W
            )

            self.pose_features = pose_features
            self.pose_sequence.append(pose_array)

            print(f"Use Rule-Based Posture: {self.use_rule_based}")
            if self.use_rule_based:
                posture_idx = classify_posture(pose_features)
                print(f"Posture Index: {posture_idx}") # This line is already there
                posture_label = postures[posture_idx][0]
                posture_color = postures[posture_idx][1]

                cv2.putText(
                    display_frame,
                    f"Posture: {posture_label}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    posture_color,
                    2,
                    cv2.LINE_AA
                )

        # Now, you have 'predicted_emotion_index' and posture information.
        # We'll combine them in the next step. For now, let's just display them.
        emotion_text = f"Face Emotion: {emotions[predicted_emotion_index][0]}" if predicted_emotion_index is not None else "Face Emotion: None"
        cv2.putText(display_frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Posture text is already displayed above

        cv2.imshow("Multimodal Analysis", display_frame)
        return display_frame

def train_multimodal_model(data_dir, output_dir, epochs=50, batch_size=16, sequence_length=10):
    """Train the multimodal model with collected data"""
    # Prepare data
    print("Preparing training data...")
    faces, poses, emotions, postures = prepare_training_data(data_dir, sequence_length)

    # Create and compile model
    model = create_multimodal_emotion_model(sequence_length=sequence_length)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'model_{epoch:02d}_{val_loss:.2f}.h5'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train the model
    print(f"Training model with {len(faces)} samples...")
    history = model.fit(
        [faces, poses],
        {
            'emotion_output': emotions,
            'posture_output': postures
        },
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Save the final model
    model.save(os.path.join(output_dir, 'multimodal_final_model.h5'))
    print(f"Model saved to {os.path.join(output_dir, 'multimodal_final_model.h5')}")

    return history

def main():
    """Main function to run the program"""
    parser = argparse.ArgumentParser(description="Multimodal Emotion and Posture Analysis")
    parser.add_argument('--mode', choices=['analyze', 'collect', 'train'], default='analyze',
                        help='Program mode: analyze (default), collect data, or train model')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model for analysis mode')
    parser.add_argument('--data-dir', type=str, default='collected_data',
                        help='Directory for data collection or training')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                        help='Directory for saving trained models')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Sequence length for temporal analysis')
    parser.add_argument('--rule-based', action='store_true',
                        help='Use rule-based posture classification')

    args = parser.parse_args()

    if args.mode == 'analyze':
        # Run real-time analysis
        analyzer = MultimodalAnalyzer(
            model_path=args.model,
            sequence_length=args.sequence_length,
            use_rule_based=args.rule_based
        )

        # Start webcam
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            # Process frame
            display_frame = analyzer.process_frame(frame)

            # Display
            cv2.imshow('Multimodal Emotion and Posture Analysis', display_frame)

            # Check for key press
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif args.mode == 'collect':
        # Run data collection mode
        collector = MultimodalDataCollector(output_dir=args.data_dir, collect_emotion=False) # Pass collect_emotion=False
        print(f"Collector Initialized: {collector}")
        print(dir(collector))  # Check available methods

        # Start webcam
        cap = cv2.VideoCapture(0)

        recording = False
        current_posture = None

        # Define keyboard shortcuts for postures
        posture_keys = {
            ord('1'): 'defensive',
            ord('2'): 'confident',
            ord('3'): 'anxious',
            ord('4'): 'attentive',
            ord('5'): 'dominant',
            ord('6'): 'neutral'
        }

        print("Data Collection Mode (Emotion Collection Disabled)")
        print("-----------------------------------------------")
        print("Press posture key (1=defensive, 2=confident, 3=anxious, 4=attentive, 5=dominant, 6=neutral)")
        print("Press 'r' to start/stop recording")
        print("Press 'q' to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break

            # Process frame
            face_results, pose_results, face_data, pose_data = collector.process_frame(frame)

            # Draw face and pose landmarks
            display_frame = frame.copy()

            if pose_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    display_frame,
                    pose_results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )

            # Display current settings and recording status
            text_y = 30
            cv2.putText(display_frame, f"Posture: {current_posture or 'None'}", (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            text_y += 30

            if recording:
                cv2.putText(display_frame, "RECORDING", (10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                text_y += 30
                cv2.putText(display_frame, f"Frames: {len(collector.sequence_buffer['poses'])}", (10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display
            cv2.imshow('Data Collection', display_frame)

            # Check for key press
            key = cv2.waitKey(5) & 0xFF

            # Handle keyboard shortcuts
            if key in posture_keys:
                current_posture = posture_keys[key]
                print(f"Selected posture: {current_posture}")
            elif key == ord('r'):
                if recording:
                    collector.stop_recording()
                    recording = False
                else:
                    if current_posture:
                        collector.start_recording(posture=current_posture) # Only pass posture
                        recording = True
                    else:
                        print("Please select a posture first")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif args.mode == 'train':
        # Train the model
        train_multimodal_model(
            args.data_dir,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length
        )

if __name__ == "__main__":
    main()