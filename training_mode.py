# training_mode.py
"""
Modified training_mode.py that addresses class imbalance, gives more emphasis to pose data,
and saves pose normalization parameters.
"""
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter # Already in your original file

# Assuming these are correctly importable from your project structure
import setup_mediapipe # Not directly used in train_model but often part of the ecosystem
import face_recognition_model as face_model # Not directly used in train_model but often part of the ecosystem
from tensorflow.keras.models import load_model # Not used for training a new model, but good to have if needed
from frame_processing import process_frame # Not directly used in train_model
from multimodal_model import emotions # Used for num_emotions_available
from data_augmentation import augment_data # Used
from state_prediction_model import create_state_prediction_model, create_state_prediction_model_simple # Used

# Define target states (ensure this is consistent across your project)
TARGET_STATES = ["Tired", "Bored", "Defensive", "Participating", "Neutral"]
NUM_TARGET_STATES = len(TARGET_STATES)

def balance_dataset(X_emotion_onehot, X_poses_np, Y_state_onehot):
    """Balance the dataset to prevent dominant class prediction"""
    Y_state_indices = np.argmax(Y_state_onehot, axis=1)
    
    class_counts = {}
    for i in range(NUM_TARGET_STATES):
        class_counts[i] = np.sum(Y_state_indices == i)
    
    print("Original class distribution:", {TARGET_STATES[k]: v for k, v in class_counts.items() if k < len(TARGET_STATES)}) # Guard against out-of-bounds
    
    # Find the max count among classes that actually have samples
    valid_counts = [count for count in class_counts.values() if count > 0]
    if not valid_counts:
        print("Warning: No samples in any class for balancing.")
        return X_emotion_onehot, X_poses_np, Y_state_onehot # Return original if no data
        
    max_samples = max(valid_counts) if valid_counts else 0
    if max_samples == 0: # if all classes have 0 samples that are valid
        print("Warning: Max samples is 0, no data to balance effectively.")
        return X_emotion_onehot, X_poses_np, Y_state_onehot

    balanced_emotions = []
    balanced_poses = []
    balanced_states = []

    for class_idx in range(NUM_TARGET_STATES):
        indices = np.where(Y_state_indices == class_idx)[0]
        
        if len(indices) == 0:
            print(f"WARNING: No samples for class {TARGET_STATES[class_idx]} during balancing!")
            continue # Skip if no original samples
            
        balanced_emotions.append(X_emotion_onehot[indices])
        balanced_poses.append(X_poses_np[indices])
        balanced_states.append(Y_state_onehot[indices])
        
        samples_needed = max_samples - len(indices)
        if samples_needed > 0:
            extra_indices = np.random.choice(indices, size=samples_needed, replace=True)
            balanced_emotions.append(X_emotion_onehot[extra_indices])
            balanced_poses.append(X_poses_np[extra_indices])
            balanced_states.append(Y_state_onehot[extra_indices])
    
    if not balanced_states: # If no classes had any samples
        print("No data to concatenate after balancing attempt.")
        return X_emotion_onehot, X_poses_np, Y_state_onehot


    X_emotion_balanced = np.vstack(balanced_emotions)
    X_poses_balanced = np.vstack(balanced_poses)
    Y_state_balanced = np.vstack(balanced_states)
    
    balanced_indices = np.argmax(Y_state_balanced, axis=1)
    balanced_counts = {}
    for i in range(NUM_TARGET_STATES):
        balanced_counts[i] = np.sum(balanced_indices == i)
    
    print("Balanced class distribution:", {TARGET_STATES[k]: v for k, v in balanced_counts.items() if k < len(TARGET_STATES)})
    print(f"Data balanced: {len(Y_state_indices)} original samples → {len(X_emotion_balanced)} balanced samples")
    
    return X_emotion_balanced, X_poses_balanced, Y_state_balanced

def enhance_pose_data(X_poses_np_input):
    """
    Enhance pose data feature importance by extracting additional features.
    Input X_poses_np_input contains the base 99 pose features per sample.
    """
    print("Enhancing pose features...")
    num_samples = X_poses_np_input.shape[0]
    if X_poses_np_input.shape[1] != 99:
        print(f"Error: enhance_pose_data expected 99 features per sample, got {X_poses_np_input.shape[1]}")
        # Potentially raise an error or return input as is, depending on desired handling
        raise ValueError(f"enhance_pose_data expected 99 features per sample, got {X_poses_np_input.shape[1]}")

    num_landmarks = 33
    landmarks_reshaped = X_poses_np_input.reshape(num_samples, num_landmarks, 3)
    
    enhanced_features_list = []
    for i in range(num_samples):
        sample_landmarks = landmarks_reshaped[i]
        
        nose = sample_landmarks[0]
        left_shoulder = sample_landmarks[11]
        right_shoulder = sample_landmarks[12]
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        head_tilt = nose - mid_shoulder
        shoulder_slope = right_shoulder - left_shoulder
        
        enhanced_sample = np.concatenate([
            X_poses_np_input[i],      # Original 99 flattened landmarks for this sample
            head_tilt.flatten(),      # 3 features
            shoulder_slope.flatten()  # 3 features
        ])
        enhanced_features_list.append(enhanced_sample)
    
    enhanced_poses_output = np.array(enhanced_features_list)
    print(f"Pose data enhanced: {X_poses_np_input.shape[1]} original features → {enhanced_poses_output.shape[1]} enhanced features")
    return enhanced_poses_output

def train_model(args):
    print("\n=== Training Enhanced State Prediction Model (Pose-Focused) ===")
    print(f"Loading data from: {args.data_file}")
    
    X_emotion_indices = []
    X_poses_flat = [] # This will store the raw 99 features per sample
    Y_state_labels = []
    pose_dim = 33 * 3
    
    try:
        with open(args.data_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if (data.get("detected_emotion_idx") is not None and
                        data.get("pose_landmarks_flat") is not None and
                        len(data["pose_landmarks_flat"]) == pose_dim and
                        data.get("target_state") in TARGET_STATES):
                        
                        X_emotion_indices.append(data["detected_emotion_idx"])
                        X_poses_flat.append(data["pose_landmarks_flat"])
                        Y_state_labels.append(data["target_state"])
                    else:
                        print(f"Skipping invalid or incomplete data line: {line[:50]}...")
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line: {line[:50]}...")
                except Exception as e:
                    print(f"Error processing line: {line[:50]}... - {e}")
        
        print(f"Loaded {len(Y_state_labels)} valid data instances.")
        if not Y_state_labels:
            print("No valid data loaded. Cannot train.")
            return
    
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_file}")
        return
    
    label_counts_initial = Counter(Y_state_labels)
    print("Initial Class distribution from data file:", label_counts_initial)
    
    missing_states_in_data = [state for state in TARGET_STATES if label_counts_initial.get(state, 0) == 0]
    if missing_states_in_data:
        print(f"WARNING: Data file has no samples for states: {missing_states_in_data}")
        if len(missing_states_in_data) == NUM_TARGET_STATES:
            print("ERROR: No samples for any target state in the data file. Cannot train.")
            return
    
    print("Preprocessing data...")
    num_emotions_available = len(emotions)
    X_emotion_onehot = tf.keras.utils.to_categorical(X_emotion_indices, num_classes=num_emotions_available)
    
    # X_poses_np here contains the raw 99 features per sample
    X_poses_np = np.array(X_poses_flat, dtype=np.float32) 
    
    # <<< START: MODIFIED NORMALIZATION SECTION >>>
    pose_min_for_scaling = np.min(X_poses_np)
    pose_max_for_scaling = np.max(X_poses_np)
    print(f"Base pose data original range (for scaling): min={pose_min_for_scaling:.4f}, max={pose_max_for_scaling:.4f}")
    
    pose_data_was_normalized = False
    X_poses_processed_for_balancing = X_poses_np.copy() # Start with a copy

    if pose_max_for_scaling > 2.0 or pose_min_for_scaling < -2.0:
        print("Normalizing base pose data to [0,1] range...")
        if (pose_max_for_scaling - pose_min_for_scaling) != 0:
            X_poses_processed_for_balancing = (X_poses_np - pose_min_for_scaling) / (pose_max_for_scaling - pose_min_for_scaling)
            pose_data_was_normalized = True
            print(f"After normalization of base pose data: min={np.min(X_poses_processed_for_balancing):.4f}, max={np.max(X_poses_processed_for_balancing):.4f}")
        else:
            print("Warning: Pose min and max for scaling are identical, skipping normalization of base pose data.")
            # X_poses_processed_for_balancing remains a copy of X_poses_np
            # pose_data_was_normalized remains False
    
    norm_params_to_save = {
        "pose_min": float(pose_min_for_scaling),
        "pose_max": float(pose_max_for_scaling),
        "was_normalized": pose_data_was_normalized
    }
    
    base_model_path, _ = os.path.splitext(args.state_model_path)
    norm_params_path = base_model_path + '_norm_params.json'
    
    try:
        with open(norm_params_path, 'w') as f:
            json.dump(norm_params_to_save, f, indent=4)
        print(f"Normalization parameters saved to {norm_params_path}")
    except IOError as e:
        print(f"Error saving normalization parameters: {e}")
    # <<< END: MODIFIED NORMALIZATION SECTION >>>
    
    state_to_index = {label: i for i, label in enumerate(TARGET_STATES)}
    Y_state_indices = [state_to_index[label] for label in Y_state_labels if label in state_to_index]
    if not Y_state_indices:
        print("ERROR: No valid target state labels after mapping. Cannot create Y_state_onehot.")
        return
    Y_state_onehot = tf.keras.utils.to_categorical(Y_state_indices, num_classes=NUM_TARGET_STATES)
    
    # Ensure data for balancing is not empty
    if X_emotion_onehot.shape[0] == 0 or X_poses_processed_for_balancing.shape[0] == 0 or Y_state_onehot.shape[0] == 0:
        print("ERROR: Not enough data to proceed with balancing.")
        return

    X_emotion_balanced, X_poses_base_balanced, Y_state_balanced = balance_dataset(
        X_emotion_onehot, X_poses_processed_for_balancing, Y_state_onehot # Use processed (potentially normalized) base poses
    )
    
    if X_poses_base_balanced.shape[0] == 0:
        print("ERROR: Data is empty after balancing. Cannot proceed.")
        return

    # IMPORTANT: enhance_pose_data now operates on the balanced (and potentially normalized) base pose data
    X_poses_enhanced_balanced = enhance_pose_data(X_poses_base_balanced) 
    
    if args.augment:
        X_emotion_final, X_poses_final, Y_state_final = augment_data(
            X_emotion_balanced, X_poses_enhanced_balanced, Y_state_balanced, 
            augmentation_factor=args.augmentation_factor
        )
    else:
        X_emotion_final, X_poses_final, Y_state_final = X_emotion_balanced, X_poses_enhanced_balanced, Y_state_balanced
    
    print(f"Final data shapes: Emotions={X_emotion_final.shape}, "
         f"Poses={X_poses_final.shape}, States={Y_state_final.shape}")
    
    if X_poses_final.shape[0] == 0:
        print("ERROR: Final dataset is empty before model training. Cannot proceed.")
        return

    if args.simple_model:
        state_model = create_state_prediction_model_simple(
            num_emotions=num_emotions_available,
            pose_input_dim=X_poses_final.shape[1],
            num_states=NUM_TARGET_STATES
        )
    else:
        state_model = create_state_prediction_model(
            num_emotions=num_emotions_available,
            pose_input_dim=X_poses_final.shape[1],
            num_states=NUM_TARGET_STATES
        )
    
    print("Starting training...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.state_model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6, verbose=1
        )
    ]
    
    history = state_model.fit(
        [X_emotion_final, X_poses_final], Y_state_final,
        epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2,
        callbacks=callbacks, shuffle=True
    )
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    try:
        plt.savefig('training_history.png')
        print("Training history plot saved to training_history.png")
    except Exception as e:
        print(f"Error saving training history plot: {e}")
    
    print(f"Training finished. Best model saved to {args.state_model_path}")