from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, Rescaling
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import cv2
import tensorflow as tf
from multimodal_model import emotions

class VGGNet(Sequential):
    """VGGNet architecture for emotion recognition"""
    def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
        super().__init__()
        # Model layers definition
        self.add(Rescaling(1./255, input_shape=input_shape))
        
        # Block 1
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        # Block 2
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        # Block 3
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        # Block 4
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        # Output block
        self.add(Flatten())
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(256, activation='relu'))
        self.add(Dense(num_classes, activation='softmax'))

        self.checkpoint_path = checkpoint_path

    def compile_model(self, lr=1e-3):
        """Compile the model with optimizer and loss function"""
        self.compile(
            optimizer=Adam(learning_rate=lr),
            loss=categorical_crossentropy,
            metrics=['accuracy']
        )

def resize_face(face):
    """Resize face image to 48x48 for model input"""
    # Convert to grayscale if needed
    if len(face.shape) == 3 and face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    elif len(face.shape) == 3 and face.shape[2] == 1:
        face = face.squeeze(axis=-1)

    # Convert to tensor and add channel dimension
    x = tf.expand_dims(tf.convert_to_tensor(face, dtype=tf.float32), axis=2)
    
    # Resize to model input shape
    return tf.image.resize(x, (48, 48))

def recognition_preprocessing(face_tensors):
    """Stack face tensors into a batch for the emotion recognition model"""
    return tf.stack(face_tensors)

def load_emotion_models(face_model_weights_1, face_model_weights_2):
    """Load and return the facial emotion models"""
    print("Loading facial emotion models...")
    face_model_1 = VGGNet(input_shape=(48, 48, 1), num_classes=len(emotions), 
                         checkpoint_path=face_model_weights_1)
    face_model_2 = VGGNet(input_shape=(48, 48, 1), num_classes=len(emotions), 
                         checkpoint_path=face_model_weights_2)
    
    face_model_1.load_weights(face_model_1.checkpoint_path)
    face_model_2.load_weights(face_model_2.checkpoint_path)
    print("Facial emotion models loaded.")
    
    return face_model_1, face_model_2