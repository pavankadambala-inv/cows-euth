# Enhanced Human Pose Inference - Usage Guide

## Overview
The enhanced `human_pose_inference.py` script now supports:
- ✅ Frame extraction (individual images)
- ✅ CSV export (consolidated pose data)
- ✅ JSON export (hierarchical pose data)
- ✅ All original features (annotated video, text files, real-time display)

## Quick Start

### Basic Usage - Frame Extraction + CSV Output
```bash
python human_pose_inference.py \
    --source your_video.mp4 \
    --save-frames \
    --output-csv
```

### Full Feature Usage
```bash
python human_pose_inference.py \
    --source your_video.mp4 \
    --save-frames \
    --output-csv \
    --output-json \
    --save-txt \
    --show
```

## Command-Line Arguments

### Required
- `--source`: Path to input video file

### Optional - Model Paths
- `--human-detector`: Path to human detector model (default: `HumanDetector-v2.pt`)
- `--pose-model`: Path to pose model (default: `yolo11n-pose.pt`)

### Optional - Output Options
- `--output-dir`: Output directory (default: `output`)
- `--save-frames`: Save individual frames as images
- `--output-csv`: Generate consolidated CSV file with all pose data
- `--output-json`: Generate JSON file with hierarchical pose data
- `--save-txt`: Save pose keypoints to individual text files (original feature)
- `--show`: Display real-time results

### Optional - Detection Thresholds
- `--conf-human`: Confidence threshold for human detection (default: 0.5)
- `--conf-pose`: Confidence threshold for pose estimation (default: 0.25)
- `--device`: Device to run on (default: `cuda:0`, use `cpu` for CPU)

## Output Structure

When you run with `--save-frames` and `--output-csv`, you'll get:

```
output/
├── video_name_frames_TIMESTAMP/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── video_name_pose_data_TIMESTAMP.csv
├── video_name_human_pose_TIMESTAMP.avi (annotated video)
└── (optional) video_name_pose_data_TIMESTAMP.json
```

## CSV Format

The CSV file contains the following columns:

### Basic Information
- `frame_number`: Frame index (0-based)
- `timestamp_seconds`: Time in seconds from start of video
- `timestamp_formatted`: Human-readable time (HH:MM:SS.mmm)
- `human_id`: Human ID within the frame (0, 1, 2, ... for multiple people)
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: Bounding box coordinates
- `human_conf`: Human detection confidence score

### Pose Keypoints (17 COCO keypoints)
For each keypoint: `{keypoint_name}_x`, `{keypoint_name}_y`, `{keypoint_name}_conf`

Keypoints included:
- nose
- left_eye, right_eye
- left_ear, right_ear
- left_shoulder, right_shoulder
- left_elbow, right_elbow
- left_wrist, right_wrist
- left_hip, right_hip
- left_knee, right_knee
- left_ankle, right_ankle

**Total columns**: 9 (basic info + timestamps) + 51 (17 keypoints × 3 values) = 60 columns

## JSON Format

The JSON file contains:
```json
{
  "video_info": {
    "source": "path/to/video.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "total_frames": 1000,
    "processed_frames": 1000
  },
  "keypoint_names": ["nose", "left_eye", ...],
  "pose_data": [
    {
      "frame_number": 0,
      "human_id": 0,
      "bbox": [x1, y1, x2, y2],
      "human_conf": 0.95,
      "keypoints": [[x, y], [x, y], ...],
      "keypoint_conf": [0.9, 0.85, ...]
    },
    ...
  ]
}
```

## Examples

### Example 1: Just frames and CSV
```bash
python human_pose_inference.py \
    --source dance_video.mp4 \
    --save-frames \
    --output-csv
```

### Example 2: All outputs, custom confidence
```bash
python human_pose_inference.py \
    --source sports_video.mp4 \
    --save-frames \
    --output-csv \
    --output-json \
    --conf-human 0.7 \
    --conf-pose 0.3
```

### Example 3: Real-time display with CSV
```bash
python human_pose_inference.py \
    --source webcam_recording.mp4 \
    --save-frames \
    --output-csv \
    --show
```

### Example 4: CPU mode
```bash
python human_pose_inference.py \
    --source video.mp4 \
    --save-frames \
    --output-csv \
    --device cpu
```

## Processing Time

Processing time depends on:
- Video resolution and length
- Number of people in the frame
- Hardware (GPU vs CPU)
- Enabled features

**Typical speeds** (with GPU):
- 1080p video: ~15-30 FPS
- 720p video: ~30-60 FPS
- 4K video: ~5-15 FPS

## Tips

1. **Large Videos**: For very large videos, consider processing in batches or use lower confidence thresholds
2. **Multiple People**: The script handles multiple people per frame automatically
3. **Storage**: Saving frames requires significant disk space (~1-5 MB per frame for HD video)
4. **CSV Analysis**: The CSV file can be easily loaded with pandas for analysis:
   ```python
   import pandas as pd
   df = pd.read_csv('video_name_pose_data_TIMESTAMP.csv')
   ```

## Troubleshooting

- **Out of Memory**: Use `--device cpu` or process shorter videos
- **Slow Processing**: Increase confidence thresholds (`--conf-human 0.7 --conf-pose 0.4`)
- **No Poses Detected**: Lower confidence thresholds or check video quality

---

## What's New

✨ **New Features in This Version**:
- `--save-frames`: Extract frames as individual JPEG images
- `--output-csv`: Export all pose data to a single CSV file
- `--output-json`: Export pose data to structured JSON file
- Better progress tracking
- Consolidated output reports

