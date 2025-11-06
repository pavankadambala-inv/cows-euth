# Shooting Pose Detection Inference - Usage Guide

## Overview

This script combines human detection, pose estimation, and XGBoost classification to automatically identify shooting poses in videos.

## Pipeline

```
Input Video
    ↓
Human Detection (YOLO) → Detect people
    ↓
Pose Estimation (YOLO-Pose) → Extract 17 keypoints
    ↓
Feature Extraction → Create 51-element feature vector
    ↓
XGBoost Classifier → Predict shooting/non-shooting pose
    ↓
Visualization → Annotate with skeleton + labels
    ↓
Output: Annotated Video + CSV/JSON predictions
```

## Prerequisites

### Required Files

1. **Human Detector Model**: `models/HumanDetector-v2.pt`
2. **Pose Model**: `models/yolo11n-pose.pt`
3. **XGBoost Model**: `shooting_pose_xgboost_model.json` (or `.pkl`)
4. **Feature Names** (optional): `feature_names.txt`

### Python Dependencies

```bash
pip install opencv-python numpy ultralytics xgboost
```

## Basic Usage

### Minimal Command

```bash
python shooting_pose_detection_inference.py \
    --source /path/to/your/video.mp4
```

This uses all default model paths and saves output to `./output/`.

### Full Command with All Options

```bash
python shooting_pose_detection_inference.py \
    --source /path/to/your/video.mp4 \
    --human-detector models/HumanDetector-v2.pt \
    --pose-model models/yolo11n-pose.pt \
    --xgboost-model shooting_pose_xgboost_model.json \
    --feature-names feature_names.txt \
    --output-dir output \
    --conf-human 0.5 \
    --conf-pose 0.25 \
    --prediction-threshold 0.5 \
    --device cuda:0 \
    --show \
    --output-csv \
    --output-json \
    --save-txt
```

## Command Line Arguments

### Required Arguments

- `--source`: Path to input video file

### Model Arguments

- `--human-detector`: Path to human detector model (default: `models/HumanDetector-v2.pt`)
- `--pose-model`: Path to pose estimation model (default: `models/yolo11n-pose.pt`)
- `--xgboost-model`: Path to XGBoost classifier (default: `shooting_pose_xgboost_model.json`)
- `--feature-names`: Path to feature names file (optional)
- `--model-format`: XGBoost model format - `json` or `pickle` (default: `json`)

### Detection Arguments

- `--conf-human`: Confidence threshold for human detection (default: `0.5`)
- `--conf-pose`: Confidence threshold for pose estimation (default: `0.25`)
- `--prediction-threshold`: Classification threshold (default: `0.5`)

### Output Arguments

- `--output-dir`: Output directory (default: `output`)
- `--output-csv`: Save predictions to CSV (enabled by default)
- `--no-output-csv`: Disable CSV output
- `--output-json`: Save predictions to JSON file
- `--save-txt`: Save predictions to individual text files per frame

### Runtime Arguments

- `--device`: Compute device - `cuda:0`, `cuda:1`, or `cpu` (default: `cuda:0`)
- `--show`: Display real-time video preview (press 'q' to quit)

## Output Files

### 1. Annotated Video

**File**: `{video_name}_shooting_detection_{timestamp}.mp4`

**Features**:
- Bounding boxes around detected humans
- Pose skeletons overlaid on each person
- **RED** box + skeleton = SHOOTING POSE
- **GREEN** box + skeleton = NON-SHOOTING POSE
- Prediction labels with confidence scores
- Real-time statistics overlay

### 2. CSV File (default: enabled)

**File**: `{video_name}_predictions_{timestamp}.csv`

**Columns**:
- `frame_number`: Frame index
- `timestamp_seconds`: Video timestamp in seconds
- `timestamp_formatted`: Human-readable timestamp (HH:MM:SS.mmm)
- `human_id`: Person ID in frame (0, 1, 2, ...)
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2`: Bounding box coordinates
- `human_conf`: Human detection confidence
- `prediction`: Classification result (0 = non-shooting, 1 = shooting)
- `prob_non_shooting`: Probability of non-shooting pose
- `prob_shooting`: Probability of shooting pose
- 51 keypoint features: `nose_x, nose_y, nose_conf, left_eye_x, ...`

### 3. JSON File (optional)

**File**: `{video_name}_predictions_{timestamp}.json`

**Structure**:
```json
{
  "video_info": {
    "source": "path/to/video.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "total_frames": 1500
  },
  "statistics": {
    "total_detections": 1234,
    "shooting_poses": 56,
    "non_shooting_poses": 1178,
    "shooting_percentage": 4.54
  },
  "predictions": [...]
}
```

### 4. Text Files (optional)

**Directory**: `{video_name}_predictions_{timestamp}/`

Individual `.txt` file per frame with detailed predictions.

## Examples

### Example 1: Basic Detection

```bash
python shooting_pose_detection_inference.py \
    --source videos/basketball_game.mp4
```

### Example 2: With Real-Time Display

```bash
python shooting_pose_detection_inference.py \
    --source videos/basketball_game.mp4 \
    --show
```

### Example 3: High-Precision Detection

```bash
python shooting_pose_detection_inference.py \
    --source videos/basketball_game.mp4 \
    --conf-human 0.7 \
    --conf-pose 0.5 \
    --prediction-threshold 0.6
```

### Example 4: Full Output (CSV + JSON + Text)

```bash
python shooting_pose_detection_inference.py \
    --source videos/basketball_game.mp4 \
    --output-csv \
    --output-json \
    --save-txt \
    --output-dir results/game1
```

### Example 5: Using Pickle Model on CPU

```bash
python shooting_pose_detection_inference.py \
    --source videos/basketball_game.mp4 \
    --xgboost-model shooting_pose_xgboost_model.pkl \
    --model-format pickle \
    --device cpu
```

## Visualization Guide

### Color Coding

| Pose Type | Bounding Box | Skeleton | Keypoints |
|-----------|--------------|----------|-----------|
| **Shooting** | Red | Red | Magenta |
| **Non-Shooting** | Green | Blue | Red |

### Labels

- **"SHOOTING POSE (85%)"**: Detected shooting pose with 85% confidence
- **"NON-SHOOTING (92%)"**: Detected non-shooting pose with 92% confidence

### Statistics Overlay

- Top-left corner shows frame info and real-time statistics
- Yellow text displays cumulative shooting pose count

## Performance Tips

### For Faster Processing

1. Use GPU: `--device cuda:0`
2. Lower human detection confidence: `--conf-human 0.3`
3. Disable display: Remove `--show` flag
4. Disable unnecessary outputs: Use `--no-output-csv` if not needed

### For Better Accuracy

1. Increase confidence thresholds:
   ```bash
   --conf-human 0.7 --conf-pose 0.5
   ```

2. Adjust prediction threshold based on your needs:
   - Higher threshold (0.7-0.9): Fewer false positives, may miss some shooting poses
   - Lower threshold (0.3-0.5): More sensitive, may have false positives

## Troubleshooting

### Issue: "Could not open video"

**Solution**: Check video path and ensure the file exists and is readable.

### Issue: "CUDA out of memory"

**Solution**: 
1. Use CPU: `--device cpu`
2. Or process a smaller video/lower resolution

### Issue: Model not found

**Solution**: Verify all model paths are correct:
```bash
ls -lh shooting_pose_xgboost_model.json
ls -lh models/HumanDetector-v2.pt
ls -lh models/yolo11n-pose.pt
```

### Issue: Predictions seem inaccurate

**Solution**:
1. Check if the video domain matches training data
2. Verify feature extraction order matches training
3. Try adjusting `--prediction-threshold`

## Understanding the Output

### Reading CSV Data

```python
import pandas as pd

# Load predictions
df = pd.read_csv('output/video_predictions_20251106_123456.csv')

# Filter shooting poses only
shooting_poses = df[df['prediction'] == 1]

# Get shooting poses with high confidence (>80%)
high_conf_shooting = df[(df['prediction'] == 1) & (df['prob_shooting'] > 0.8)]

# Analyze by timestamp
shooting_times = shooting_poses['timestamp_formatted'].unique()
print(f"Shooting poses detected at: {shooting_times}")
```

### JSON Analysis

```python
import json

# Load JSON
with open('output/video_predictions_20251106_123456.json', 'r') as f:
    data = json.load(f)

# Get statistics
stats = data['statistics']
print(f"Total detections: {stats['total_detections']}")
print(f"Shooting poses: {stats['shooting_poses']} ({stats['shooting_percentage']:.1f}%)")
```

## Feature Vector Details

The XGBoost model expects 51 features in this exact order:

1. **Nose**: `nose_x, nose_y, nose_conf`
2. **Eyes**: `left_eye_x, left_eye_y, left_eye_conf, right_eye_x, right_eye_y, right_eye_conf`
3. **Ears**: `left_ear_x, left_ear_y, left_ear_conf, right_ear_x, right_ear_y, right_ear_conf`
4. **Shoulders**: `left_shoulder_x, left_shoulder_y, left_shoulder_conf, right_shoulder_x, right_shoulder_y, right_shoulder_conf`
5. **Elbows**: `left_elbow_x, left_elbow_y, left_elbow_conf, right_elbow_x, right_elbow_y, right_elbow_conf`
6. **Wrists**: `left_wrist_x, left_wrist_y, left_wrist_conf, right_wrist_x, right_wrist_y, right_wrist_conf`
7. **Hips**: `left_hip_x, left_hip_y, left_hip_conf, right_hip_x, right_hip_y, right_hip_conf`
8. **Knees**: `left_knee_x, left_knee_y, left_knee_conf, right_knee_x, right_knee_y, right_knee_conf`
9. **Ankles**: `left_ankle_x, left_ankle_y, left_ankle_conf, right_ankle_x, right_ankle_y, right_ankle_conf`

**Total**: 17 keypoints × 3 (x, y, confidence) = **51 features**

## Next Steps

### Batch Processing Multiple Videos

```bash
#!/bin/bash
for video in videos/*.mp4; do
    python shooting_pose_detection_inference.py \
        --source "$video" \
        --output-dir "results/$(basename "$video" .mp4)"
done
```

### Analyzing Results

After processing, you can:
1. Review annotated videos to verify predictions
2. Analyze CSV data with pandas/Excel
3. Extract shooting pose timestamps for highlight reels
4. Calculate shooting form metrics from keypoints

## Contact & Support

For issues or questions about the script, check:
- Model training notebook: `shooting_pose_xgboost_classifier.ipynb`
- Pose inference script: `human_pose_inference.py`
- Real-time visualization: `shooting_pose_realtime_viz.py`


