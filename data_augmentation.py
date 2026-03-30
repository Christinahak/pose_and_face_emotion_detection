import numpy as np

def augment_data(X_emotion_onehot, X_poses_np, Y_state_onehot, augmentation_factor=3):
    """
    Augment the training data by applying various transformations to pose data
    and slight variations to emotion probabilities.
    
    Args:
        X_emotion_onehot: One-hot encoded emotion data
        X_poses_np: Pose landmark data (flattened)
        Y_state_onehot: One-hot encoded target states
        augmentation_factor: How many augmented samples to create per original sample
    
    Returns:
        Augmented datasets (emotions, poses, states)
    """
    print(f"Augmenting data (factor: {augmentation_factor})...")
    num_samples = X_emotion_onehot.shape[0]
    augmented_emotions = [X_emotion_onehot]
    augmented_poses = [X_poses_np]
    augmented_states = [Y_state_onehot]
    
    for _ in range(augmentation_factor - 1):  # -1 because we already have the original data
        # 1. Augment poses by adding small random variations
        pose_noise = np.random.normal(0, 0.02, X_poses_np.shape)  # Small Gaussian noise
        augmented_pose = X_poses_np + pose_noise
        
        # Ensure the augmented poses maintain reasonable values
        # Assuming poses are normalized to [0,1] range
        augmented_pose = np.clip(augmented_pose, 0.0, 1.0)
        
        # 2. Slightly modify emotion probabilities (subtle changes to confidence)
        # This simulates slight variations in facial expressions
        augmented_emotion = X_emotion_onehot.copy()
        for i in range(num_samples):
            if np.max(augmented_emotion[i]) > 0:  # If there's a valid emotion
                # Get the index of the highest probability emotion
                max_idx = np.argmax(augmented_emotion[i])
                
                # Add small random variation to the probabilities
                # (keeping the same primary emotion but with different confidence)
                noise = np.random.uniform(-0.1, 0.1, augmented_emotion[i].shape)
                augmented_emotion[i] += noise
                augmented_emotion[i] = np.clip(augmented_emotion[i], 0.0, 1.0)
                
                # Ensure the max emotion remains the same by boosting it if needed
                if np.argmax(augmented_emotion[i]) != max_idx:
                    augmented_emotion[i][max_idx] = np.max(augmented_emotion[i]) + 0.05
                
                # Normalize to ensure it still sums to 1
                augmented_emotion[i] /= np.sum(augmented_emotion[i])
        
        # 3. Keep the same state labels for the augmented data
        augmented_states.append(Y_state_onehot)
        augmented_emotions.append(augmented_emotion)
        augmented_poses.append(augmented_pose)
    
    # Concatenate all augmented data
    X_emotion_aug = np.vstack(augmented_emotions)
    X_poses_aug = np.vstack(augmented_poses)
    Y_state_aug = np.vstack(augmented_states)
    
    print(f"Data augmented: {num_samples} original samples → {X_emotion_aug.shape[0]} total samples")
    return X_emotion_aug, X_poses_aug, Y_state_aug
