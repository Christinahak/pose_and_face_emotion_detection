import tensorflow as tf

def create_state_prediction_model(num_emotions, pose_input_dim, num_states):
    """Create model to predict student state from emotion and pose with more weight on pose features"""
    # Input layers
    emotion_input = tf.keras.layers.Input(shape=(num_emotions,), name='emotion_input_onehot')
    pose_input = tf.keras.layers.Input(shape=(pose_input_dim,), name='pose_input_flat')
    
    # Process pose features - ENHANCED with more layers and neurons
    pose_features = tf.keras.layers.Dense(256, activation='relu')(pose_input)  # Increased from 128
    pose_features = tf.keras.layers.BatchNormalization()(pose_features)
    pose_features = tf.keras.layers.Dropout(0.3)(pose_features)  # Reduced dropout for more influence
    
    pose_features = tf.keras.layers.Dense(128, activation='relu')(pose_features)  # Increased from 64
    pose_features = tf.keras.layers.BatchNormalization()(pose_features)
    pose_features = tf.keras.layers.Dropout(0.3)(pose_features)
    
    # Additional pose processing layer
    pose_features = tf.keras.layers.Dense(64, activation='relu')(pose_features)
    pose_features = tf.keras.layers.BatchNormalization()(pose_features)
    
    # Process emotion features - REDUCED importance
    emotion_features = tf.keras.layers.Dense(16, activation='relu')(emotion_input)  # Reduced from 32
    emotion_features = tf.keras.layers.Dropout(0.5)(emotion_features)  # Added dropout to reduce influence
    
    # Combine features with attention mechanism to weigh pose more heavily
    combined = tf.keras.layers.Concatenate()([emotion_features, pose_features])
    
    # Add attention layer to focus on pose features
    attention = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
    emotion_weighted = tf.keras.layers.Multiply()([emotion_features, 1 - attention])  # Less weight for emotion
    pose_weighted = tf.keras.layers.Multiply()([pose_features, attention])  # More weight for pose
    
    # Recombine with weighted features
    combined = tf.keras.layers.Concatenate()([emotion_weighted, pose_weighted])
    
    # Additional processing
    combined = tf.keras.layers.Dense(128, activation='relu')(combined)
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dropout(0.4)(combined)  # Reduced dropout
    combined = tf.keras.layers.Dense(64, activation='relu')(combined)
    
    # Output layer
    output = tf.keras.layers.Dense(num_states, activation='softmax', name='state_output')(combined)
    
    # Create and compile model
    model = tf.keras.models.Model(inputs=[emotion_input, pose_input], outputs=output, name="StatePredictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Enhanced State Prediction Model Compiled (Pose-Focused).")
    model.summary()
    return model

# Alternative approach if the attention mechanism is too complex
def create_state_prediction_model_simple(num_emotions, pose_input_dim, num_states):
    """Create model with simpler approach to give more weight to pose features"""
    # Input layers
    emotion_input = tf.keras.layers.Input(shape=(num_emotions,), name='emotion_input_onehot')
    pose_input = tf.keras.layers.Input(shape=(pose_input_dim,), name='pose_input_flat')
    
    # Process pose features - ENHANCED with more capacity
    pose_features = tf.keras.layers.Dense(256, activation='relu')(pose_input)
    pose_features = tf.keras.layers.BatchNormalization()(pose_features)
    pose_features = tf.keras.layers.Dropout(0.3)(pose_features)
    
    pose_features = tf.keras.layers.Dense(128, activation='relu')(pose_features)
    pose_features = tf.keras.layers.BatchNormalization()(pose_features)
    pose_features = tf.keras.layers.Dropout(0.3)(pose_features)
    
    # Additional pose processing layer
    pose_features = tf.keras.layers.Dense(64, activation='relu')(pose_features)
    pose_features = tf.keras.layers.BatchNormalization()(pose_features)
    
    # Process emotion features - REDUCED importance
    emotion_features = tf.keras.layers.Dense(16, activation='relu')(emotion_input)
    emotion_features = tf.keras.layers.Dropout(0.5)(emotion_features)  # More dropout to reduce influence
    
    # Combine features with different proportions
    # The pose features vector is larger (64 neurons vs 16 for emotion)
    # so it will naturally have more influence in the concatenated result
    combined = tf.keras.layers.Concatenate()([emotion_features, pose_features])
    
    # Processing with regularization
    combined = tf.keras.layers.Dense(128, activation='relu')(combined)
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dropout(0.4)(combined)
    combined = tf.keras.layers.Dense(64, activation='relu')(combined)
    
    # Output layer
    output = tf.keras.layers.Dense(num_states, activation='softmax', name='state_output')(combined)
    
    # Create and compile model
    model = tf.keras.models.Model(inputs=[emotion_input, pose_input], outputs=output, name="StatePredictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Enhanced State Prediction Model Compiled (Simpler Pose-Focused).")
    model.summary()
    return model