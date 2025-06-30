# Blink Rate Monitoring using YOLOv8 and MediaPipe Face Mesh

Real-time blink detection and BPM calculation with live visualization.

## Features

- **Real-time blink detection** using webcam feed
- **YOLOv8n** for fast and accurate face detection
- **MediaPipe Face Mesh** for precise eye landmark extraction
- **Eye Aspect Ratio (EAR)** calculation for blink detection
- **Live BPM tracking** with 60-second rolling window
- **Real-time visualization** with separate graph window
- **Performance metrics** display (FPS, EAR, BPM)

## Installation

1. Clone or download this project
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the blink detector:

```bash
python blink_detector.py
```

### Controls

- **'q'**: Quit the application
- **'r'**: Reset blink counter

### Display Information

The main camera window shows:
- Live video feed with face detection box
- Eye landmarks (yellow dots)
- Real-time metrics overlay:
  - FPS: Current frames per second
  - Blinks: Total blinks counted
  - BPM: Blinks per minute (60-second window)
  - EAR: Current Eye Aspect Ratio
  - Threshold: EAR threshold for blink detection

The separate graph window displays:
- **Top graph**: Live BPM trend over time
- **Bottom graph**: Live EAR values with threshold line

## How It Works

### 1. Face Detection (YOLOv8n)
- Detects faces in the camera feed
- Crops the largest face region for processing
- Prioritizes the first detected face
- Handles temporary detection failures gracefully

### 2. Eye Landmark Extraction (MediaPipe)
- Extracts 468 facial landmarks from cropped face
- Focuses on 6 key points per eye for EAR calculation
- **Left eye indices**: 362, 382, 381, 380, 374, 373
- **Right eye indices**: 33, 7, 163, 144, 145, 153

### 3. Blink Detection Logic
- **Eye Aspect Ratio (EAR)**: `(||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`
- **Threshold**: 0.21 (configurable)
- **Open eye**: EAR ≈ 0.25-0.35
- **Closed eye**: EAR ≈ 0.10-0.15
- Blink counted when EAR drops below threshold and returns above

### 4. BPM Calculation
- Tracks blink timestamps
- Calculates blinks per minute using 60-second rolling window
- Updates in real-time

## Technical Specifications

- **YOLOv8n**: Lightweight object detection for face detection
- **MediaPipe Face Mesh**: 468-point facial landmark detection
- **EAR Threshold**: 0.21 (optimized for most users)
- **Minimum Blink Duration**: 2 consecutive frames
- **Rolling Window**: 60 seconds for BPM calculation
- **Update Rate**: Live plotting updates every 10 frames

## Error Handling

- **No face detected**: Displays warning message, continues running
- **Multiple faces**: Prioritizes largest/closest face
- **Camera access issues**: Exits with error message
- **Temporary detection failures**: Uses fallback mechanisms

## Use Cases

- **Eye strain monitoring** during prolonged screen time
- **Fatigue assessment** for drivers or workers
- **Medical applications** for blink rate analysis
- **Research** in computer vision and human behavior

## Performance Notes

- Optimized for real-time processing
- Uses YOLOv8n for speed over accuracy
- Separate thread for plotting to avoid blocking main loop
- Configurable parameters for different use cases

## Future Enhancements

- Export to ONNX/TensorRT for edge deployment
- Adaptive EAR threshold based on user baseline
- Historical data logging and analysis
- Multi-user support and face tracking
- Jetson Nano optimization
