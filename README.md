# Pose and Face Emotion Detection

A real-time multimodal system that combines **facial emotion recognition** and **body pose estimation** to predict student engagement states via webcam.

---

## What It Does

The system analyzes live webcam input across two modalities:

1. **Facial Emotion Recognition** — detects 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using an ensemble of two VGGNet models
2. **Body Pose Estimation** — tracks 33 body landmarks using MediaPipe
3. **Student State Prediction** — fuses emotion + pose data to classify one of 5 student engagement states:
   - **Tired** — low energy
   - **Bored** — disengaged
   - **Defensive** — closed body language
   - **Participating** — actively engaged
   - **Neutral** — baseline

---

## Tech Stack

- **MediaPipe** — face detection and 33-point body pose estimation
- **TensorFlow / Keras** — state prediction neural network (attention-based multi-input model)
- **OpenCV** — webcam capture and frame rendering
- **Pre-trained VGGNet** — ensemble emotion recognition from 48×48 face crops

---

## Project Structure

```
├── main.py                        # CLI entry point (analyze / collect / train modes)
├── frame_processing.py            # Per-frame face + pose detection pipeline
├── face_recognition_model.py      # VGGNet architecture for emotion recognition
├── setup_mediapipe.py             # MediaPipe initialization
├── multimodal_model.py            # Emotion class definitions and state labels
├── collection_mode.py             # Interactive webcam data labeling tool
├── training_mode.py               # Model training with class balancing + augmentation
├── state_prediction_model.py      # Neural network architectures (attention & simple)
├── analyze_mode.py                # Real-time inference and visualization
├── data_augmentation.py           # Noise injection / probability variation augmentation
├── collected_state_data.jsonl     # Collected training samples (emotion + pose + label)
├── trained_models/                # Saved .keras models and normalization parameters
└── saved_models/                  # Pre-trained VGGNet weights (vggnet.h5, vggnet_up.h5)
```

---

## How to Run

### 1. Collect Training Data

Open webcam and manually label frames in real time:

```bash
python main.py --mode collect \
  --data-file collected_state_data.jsonl \
  --face-model-weights-1 saved_models/vggnet.h5 \
  --face-model-weights-2 saved_models/vggnet_up.h5
```

Press **t / b / d / p / n** to label the current frame as Tired / Bored / Defensive / Participating / Neutral. Press **q** to quit.

### 2. Train the State Predictor

```bash
python main.py --mode train \
  --data-file collected_state_data.jsonl \
  --state-model-path trained_models/my_state_predictor.keras \
  --epochs 50 --batch-size 32 --augment --augmentation-factor 3
```

Outputs a `.keras` model, a normalization parameters JSON, and a `training_history.png` plot.

### 3. Run Real-Time Analysis


```bash
python main.py --mode analyze \
  --state-model-path trained_models/my_state_predictor.keras \
  --face-model-weights-1 saved_models/vggnet.h5 \
  --face-model-weights-2 saved_models/vggnet_up.h5
```

Displays live webcam feed with detected emotion, pose landmarks, and predicted student state. Press **ESC** to exit.

---

## Model Architecture

The state predictor is a multi-input neural network:

- **Emotion branch** — 7-dimensional one-hot vector
- **Pose branch** — 99-dimensional flattened landmarks (33 landmarks × x/y/z), optionally extended to 105 dimensions with derived features (head tilt, shoulder slope)
- **Attention mechanism** — weights pose features more heavily than emotion for state prediction
- **Output** — softmax over 5 student states

Training includes class balancing (resampling) and optional data augmentation (noise injection, probability variation) to handle imbalanced collected data.

## Example Usage

[example_display.webm](https://github.com/user-attachments/assets/9cff5a4d-1ea0-4767-a05e-84eba1950979)
