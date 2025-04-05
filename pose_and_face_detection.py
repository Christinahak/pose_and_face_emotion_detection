import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Rescaling
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Define emotions dictionary
emotions = {
    0: ['Angry', (0,0,255), (255,255,255)],
    1: ['Disgust', (0,102,0), (255,255,255)],
    2: ['Fear', (255,255,153), (0,51,51)],
    3: ['Happy', (153,0,153), (255,255,255)],
    4: ['Sad', (255,0,0), (255,255,255)],
    5: ['Surprise', (0,255,0), (255,255,255)],
    6: ['Neutral', (160,160,160), (255,255,255)]
}

# Model parameters
num_classes = len(emotions)
input_shape = (48, 48, 1)
weights_1 = 'saved_models/vggnet.h5'
weights_2 = 'saved_models/vggnet_up.h5'

# Define the VGGNet model class
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
    x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
    return tf.image.resize(x, (48,48))

def recognition_preprocessing(faces):
    x = tf.convert_to_tensor([resize_face(f) for f in faces])
    return x

def main():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing_face = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing_pose = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # 0=Lite, 1=Full, 2=Heavy
    )

    # Load emotion recognition models
    model_1 = VGGNet(input_shape, num_classes, weights_1)
    model_2 = VGGNet(input_shape, num_classes, weights_2)
    model_1.load_weights(model_1.checkpoint_path)
    model_2.load_weights(model_2.checkpoint_path)

    # Start webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        H, W, _ = image.shape
        
        # Make a copy for pose detection
        pose_image = image.copy()
        
        # Process for face detection and emotion recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb_image)
        
        # Process for pose detection
        pose_results = pose.process(rgb_image)

        # Draw pose landmarks on the pose image
        mp_drawing_pose.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Face detection and emotion recognition
        if face_results.detections:
            faces = []
            pos = []
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

                face = image[y1:y2, x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                faces.append(face)
                pos.append((x1, y1, x2, y2))

            if faces:
                x = recognition_preprocessing(faces)
                y_1 = model_1.predict(x, verbose=0)
                y_2 = model_2.predict(x, verbose=0)
                l = np.argmax(y_1 + y_2, axis=1)

                for i in range(len(faces)):
                    cv2.rectangle(image, (pos[i][0],pos[i][1]),
                                (pos[i][2],pos[i][3]), emotions[l[i]][1], 2, lineType=cv2.LINE_AA)
                    
                    cv2.rectangle(image, (pos[i][0],pos[i][1]-20),
                                (pos[i][2]+20,pos[i][1]), emotions[l[i]][1], -1, lineType=cv2.LINE_AA)
                    
                    cv2.putText(image, f'{emotions[l[i]][0]}', (pos[i][0],pos[i][1]-5),
                                0, 0.6, emotions[l[i]][2], 2, lineType=cv2.LINE_AA)
        
        # Arms crossing detection
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Get required landmarks for arms crossing detection
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Check visibility/confidence
            min_visibility = 0.7
            if (left_shoulder.visibility >= min_visibility and
                right_shoulder.visibility >= min_visibility and
                left_elbow.visibility >= min_visibility and
                right_elbow.visibility >= min_visibility and
                left_wrist.visibility >= min_visibility and
                right_wrist.visibility >= min_visibility):
                
                # Get image dimensions for converting normalized coordinates
                image_height, image_width, _ = image.shape
                
                # Convert normalized coordinates to pixel coordinates
                left_shoulder_x = int(left_shoulder.x * image_width)
                right_shoulder_x = int(right_shoulder.x * image_width)
                left_shoulder_y = int(left_shoulder.y * image_height)
                right_shoulder_y = int(right_shoulder.y * image_height)
                
                left_elbow_x = int(left_elbow.x * image_width)
                right_elbow_x = int(right_elbow.x * image_width)
                
                left_wrist_x = int(left_wrist.x * image_width)
                right_wrist_x = int(right_wrist.x * image_width)
                left_wrist_y = int(left_wrist.y * image_height)
                right_wrist_y = int(right_wrist.y * image_height)
                
                # Calculate the midpoint between shoulders
                shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2
                
                # Calculate y-range for chest area (between shoulders and a bit lower)
                chest_top_y = max(left_shoulder_y, right_shoulder_y)
                chest_bottom_y = chest_top_y + 150  # Approximate chest height
                
                # Check if arms are at chest level
                wrists_at_chest_level = (
                    chest_top_y < left_wrist_y < chest_bottom_y and
                    chest_top_y < right_wrist_y < chest_bottom_y
                )
                
                # Check if arms have crossed the midline
                arms_crossed_midline = (
                    right_wrist_x < shoulder_mid_x and
                    left_wrist_x > shoulder_mid_x
                )
                
                # Check if wrists are in opposite regions
                wrists_crossed = (
                    right_wrist_x < left_elbow_x and
                    left_wrist_x > right_elbow_x
                )
                
                # Combined condition for arm crossing
                arms_crossed = wrists_at_chest_level and (arms_crossed_midline or wrists_crossed)
                
                if arms_crossed:
                    cv2.putText(
                        image,
                        "Arms Crossed!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

        # Display the combined results
        cv2.imshow('Emotion and Posture Analysis', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()