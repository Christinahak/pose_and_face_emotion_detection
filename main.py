import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

import os
import time
import argparse
import json

# Import emotion dictionary from multimodal_model
from multimodal_model import emotions
import face_recognition_model as face_model
from analyze_mode import analyze_state
from training_mode import train_model  # Use our modified training module
from collection_mode import collect_data

def main():
    """Main entry point with argument parsing and mode selection"""
    parser = argparse.ArgumentParser(description="Multimodal State Analysis with Enhanced Pose Focus")
    parser.add_argument('--mode', choices=['analyze', 'collect', 'train'], required=True, 
                      help='Program mode')
    parser.add_argument('--data-file', type=str, default='collected_state_data.jsonl', 
                      help='File for storing/reading collected data (JSON Lines format)')
    parser.add_argument('--face-model-weights-1', type=str, default='saved_models/vggnet.h5', 
                      help='Path to first VGGNet weights')
    parser.add_argument('--face-model-weights-2', type=str, default='saved_models/vggnet_up.h5', 
                      help='Path to second VGGNet weights')
    parser.add_argument('--state-model-path', type=str, default='trained_models/my_state_predictor_augmented.keras', 
                      help='Path to save/load the trained state prediction model')
    parser.add_argument('--epochs', type=int, default=50, 
                      help='Number of epochs for training state model')
    parser.add_argument('--batch-size', type=int, default=32, 
                      help='Batch size for training state model')
    parser.add_argument('--augment', action='store_true',
                      help='Enable data augmentation during training')
    parser.add_argument('--augmentation-factor', type=int, default=3,
                      help='How many times to augment the original dataset')
    parser.add_argument('--simple-model', action='store_true',
                      help='Use simpler model architecture instead of attention mechanism')
    
    args = parser.parse_args()
    
    # Create output directory for state model
    os.makedirs(os.path.dirname(args.state_model_path), exist_ok=True)
    
    # Run selected mode
    if args.mode == 'collect':
        collect_data(args)
    elif args.mode == 'train':
        train_model(args)
    elif args.mode == 'analyze':
        analyze_state(args)

if __name__ == "__main__":
    # Import MediaPipe modules at the module level for use in functions
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    main()