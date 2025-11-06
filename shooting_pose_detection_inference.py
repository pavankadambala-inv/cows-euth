"""
Shooting Pose Detection Inference Script
Combines human detection, pose estimation, and XGBoost classification to identify shooting poses in videos.

Pipeline:
1. Human Detection (YOLO) → Detect people in frames
2. Pose Estimation (YOLO Pose) → Extract 17 COCO keypoints
3. Feature Extraction → Convert keypoints to 51-element feature vector
4. XGBoost Classification → Predict shooting pose (0/1) with probability
5. Visualization → Annotate frame with skeleton + prediction labels
6. Export → Save annotated video + CSV with predictions
"""

import cv2
import numpy as np
from ultralytics import YOLO
import xgboost as xgb
import argparse
from pathlib import Path
from datetime import datetime
import csv
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


# ==================== MODEL LOADER ====================

def load_xgboost_model(model_path, model_format='json'):
    """
    Load the trained XGBoost model.
    
    Args:
        model_path: Path to the saved model file
        model_format: 'json' or 'pickle'
    
    Returns:
        Loaded XGBoost model
    """
    print(f"Loading XGBoost model from: {model_path}")
    
    if model_format == 'json':
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    elif model_format == 'pickle':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")
    
    print("✓ XGBoost model loaded successfully!")
    return model


def load_feature_names(feature_names_path):
    """
    Load feature names to ensure correct feature order.
    
    Args:
        feature_names_path: Path to feature_names.txt
    
    Returns:
        List of feature names in correct order
    """
    feature_names = []
    with open(feature_names_path, 'r') as f:
        for line in f:
            feature_names.append(line.strip())
    
    print(f"✓ Loaded {len(feature_names)} feature names")
    return feature_names


# ==================== FEATURE EXTRACTION ====================

def extract_features_from_keypoints(keypoints, keypoint_conf):
    """
    Extract 51 features from pose keypoints in the correct order for XGBoost model.
    
    Features: 17 keypoints × 3 (x, y, confidence) = 51 features
    Order: nose, left_eye, right_eye, left_ear, right_ear,
           left_shoulder, right_shoulder, left_elbow, right_elbow,
           left_wrist, right_wrist, left_hip, right_hip,
           left_knee, right_knee, left_ankle, right_ankle
    
    Args:
        keypoints: numpy array of shape (17, 2) with x, y coordinates
        keypoint_conf: numpy array of shape (17,) with confidence scores
    
    Returns:
        numpy array of shape (51,) with features in correct order
    """
    features = []
    
    # Extract features in order: x, y, conf for each keypoint
    for i in range(17):
        features.append(keypoints[i][0])  # x coordinate
        features.append(keypoints[i][1])  # y coordinate
        features.append(keypoint_conf[i])  # confidence
    
    return np.array(features)


# ==================== MAIN INFERENCE FUNCTION ====================

def run_shooting_pose_detection(
    video_path,
    human_detector_path,
    pose_model_path,
    xgboost_model_path,
    output_dir="output",
    conf_human=0.5,
    conf_pose=0.25,
    prediction_threshold=0.5,
    device="cuda:0",
    show=False,
    save_txt=False,
    output_csv=True,
    output_json=False,
    model_format='json',
    feature_names_path=None
):
    """
    Run human detection + pose estimation + shooting pose classification pipeline.
    
    Args:
        video_path: Path to input video
        human_detector_path: Path to human detector model
        pose_model_path: Path to pose estimation model
        xgboost_model_path: Path to XGBoost shooting pose classifier
        output_dir: Output directory for results
        conf_human: Confidence threshold for human detection
        conf_pose: Confidence threshold for pose estimation
        prediction_threshold: Threshold for binary classification (default: 0.5)
        device: Device to run on ('cuda:0' or 'cpu')
        show: Show real-time results
        save_txt: Save predictions to text files
        output_csv: Save predictions to CSV file
        output_json: Save predictions to JSON file
        model_format: 'json' or 'pickle' for XGBoost model
        feature_names_path: Optional path to feature_names.txt for verification
    """
    
    # Load models
    print(f"Loading human detector from: {human_detector_path}")
    human_detector = YOLO(human_detector_path)
    
    print(f"Loading pose model from: {pose_model_path}")
    pose_model = YOLO(pose_model_path)
    
    print(f"Loading XGBoost classifier...")
    xgboost_model = load_xgboost_model(xgboost_model_path, model_format)
    
    # Optionally load and verify feature names
    if feature_names_path and Path(feature_names_path).exists():
        feature_names = load_feature_names(feature_names_path)
    else:
        # Default feature names in correct order
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        feature_names = []
        for kp in keypoint_names:
            feature_names.extend([f'{kp}_x', f'{kp}_y', f'{kp}_conf'])
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem
    output_video_path = output_path / f"{video_name}_shooting_detection_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Setup text output if needed
    if save_txt:
        txt_output_dir = output_path / f"{video_name}_predictions_{timestamp}"
        txt_output_dir.mkdir(exist_ok=True)
    
    # Collect all prediction data for CSV/JSON export
    all_prediction_data = []
    
    # COCO keypoint names for reference
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Define COCO keypoint connections for skeleton drawing
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    frame_idx = 0
    shooting_pose_count = 0
    total_detections = 0
    
    print(f"\nProcessing video with shooting pose detection...")
    print(f"Output will be saved to: {output_video_path}")
    print("="*60)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create a copy for annotation
            annotated_frame = frame.copy()
            
            # Step 1: Detect humans in the frame
            human_results = human_detector.predict(
                frame,
                conf=conf_human,
                device=device,
                verbose=False
            )[0]
            
            # Store all predictions for this frame
            frame_predictions = []
            
            # Step 2: For each detected human, run pose estimation and classification
            if len(human_results.boxes) > 0:
                for idx, box in enumerate(human_results.boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    human_conf = float(box.conf[0])
                    
                    # Ensure coordinates are within frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # Skip if box is too small
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue
                    
                    # Crop the human region with some padding
                    padding = 10
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(width, x2 + padding)
                    crop_y2 = min(height, y2 + padding)
                    
                    cropped_human = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    if cropped_human.size == 0:
                        continue
                    
                    # Run pose estimation on cropped region
                    pose_results = pose_model.predict(
                        cropped_human,
                        conf=conf_pose,
                        device=device,
                        verbose=False
                    )[0]
                    
                    # Process pose keypoints and classify
                    if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
                        # Get the first (most confident) pose
                        keypoints = pose_results.keypoints[0].xy.cpu().numpy()[0]  # Shape: (17, 2)
                        keypoint_conf = pose_results.keypoints[0].conf.cpu().numpy()[0]  # Shape: (17,)
                        
                        # Transform keypoints back to original frame coordinates
                        keypoints[:, 0] += crop_x1
                        keypoints[:, 1] += crop_y1
                        
                        # Extract features for XGBoost model
                        features = extract_features_from_keypoints(keypoints, keypoint_conf)
                        features_reshaped = features.reshape(1, -1)  # Shape: (1, 51)
                        
                        # Make prediction with XGBoost
                        prediction = xgboost_model.predict(features_reshaped)[0]
                        prediction_proba = xgboost_model.predict_proba(features_reshaped)[0]
                        
                        prob_non_shooting = prediction_proba[0]
                        prob_shooting = prediction_proba[1]
                        
                        # Count detections
                        total_detections += 1
                        if prediction == 1:
                            shooting_pose_count += 1
                        
                        # Determine colors based on prediction
                        if prediction == 1:  # Shooting pose
                            bbox_color = (0, 0, 255)  # Red
                            skeleton_color = (0, 0, 255)  # Red
                            keypoint_color = (255, 0, 255)  # Magenta
                            label_text = "SHOOTING POSE"
                            label_bg_color = (0, 0, 255)  # Red background
                        else:  # Non-shooting pose
                            bbox_color = (0, 255, 0)  # Green
                            skeleton_color = (255, 0, 0)  # Blue
                            keypoint_color = (0, 0, 255)  # Red
                            label_text = "NON-SHOOTING"
                            label_bg_color = (0, 255, 0)  # Green background
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bbox_color, 3)
                        
                        # Draw prediction label with background
                        label_full = f"{label_text} ({prob_shooting:.2%})"
                        label_size, _ = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        label_w, label_h = label_size
                        
                        # Draw label background
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - label_h - 20),
                            (x1 + label_w + 10, y1),
                            label_bg_color,
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated_frame,
                            label_full,
                            (x1 + 5, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),  # White text
                            2
                        )
                        
                        # Draw skeleton
                        for connection in skeleton_connections:
                            pt1_idx, pt2_idx = connection
                            if (keypoint_conf[pt1_idx] > 0.5 and 
                                keypoint_conf[pt2_idx] > 0.5):
                                pt1 = tuple(map(int, keypoints[pt1_idx]))
                                pt2 = tuple(map(int, keypoints[pt2_idx]))
                                cv2.line(annotated_frame, pt1, pt2, skeleton_color, 2)
                        
                        # Draw keypoints
                        for kp_idx, (kp, kp_conf) in enumerate(zip(keypoints, keypoint_conf)):
                            if kp_conf > 0.5:
                                x, y = map(int, kp)
                                cv2.circle(annotated_frame, (x, y), 5, keypoint_color, -1)
                                cv2.circle(annotated_frame, (x, y), 5, (255, 255, 255), 1)
                        
                        # Store prediction data
                        prediction_data = {
                            'frame_number': frame_idx,
                            'human_id': idx,
                            'bbox': [x1, y1, x2, y2],
                            'human_conf': human_conf,
                            'prediction': int(prediction),
                            'prob_non_shooting': float(prob_non_shooting),
                            'prob_shooting': float(prob_shooting),
                            'keypoints': keypoints.tolist(),
                            'keypoint_conf': keypoint_conf.tolist(),
                            'features': features.tolist()
                        }
                        frame_predictions.append(prediction_data)
                        
                        # Collect for CSV/JSON export
                        if output_csv or output_json:
                            timestamp_seconds = frame_idx / fps if fps > 0 else 0
                            
                            # Create entry with all data
                            entry = {
                                'frame_number': frame_idx,
                                'timestamp_seconds': timestamp_seconds,
                                'human_id': idx,
                                'bbox_x1': x1,
                                'bbox_y1': y1,
                                'bbox_x2': x2,
                                'bbox_y2': y2,
                                'human_conf': human_conf,
                                'prediction': int(prediction),
                                'prob_non_shooting': float(prob_non_shooting),
                                'prob_shooting': float(prob_shooting)
                            }
                            
                            # Add keypoint features
                            for feat_idx, feat_name in enumerate(feature_names):
                                entry[feat_name] = float(features[feat_idx])
                            
                            all_prediction_data.append(entry)
            
            # Save prediction data to text file if requested
            if save_txt and frame_predictions:
                txt_file = txt_output_dir / f"frame_{frame_idx:06d}.txt"
                with open(txt_file, 'w') as f:
                    f.write(f"Frame: {frame_idx}\n")
                    f.write(f"Timestamp: {frame_idx / fps:.3f}s\n\n")
                    for pred in frame_predictions:
                        f.write(f"\nHuman {pred['human_id']}:\n")
                        f.write(f"  BBox: {pred['bbox']}\n")
                        f.write(f"  Human Confidence: {pred['human_conf']:.3f}\n")
                        f.write(f"  Prediction: {'SHOOTING' if pred['prediction'] == 1 else 'NON-SHOOTING'}\n")
                        f.write(f"  Probability (Non-Shooting): {pred['prob_non_shooting']:.4f}\n")
                        f.write(f"  Probability (Shooting): {pred['prob_shooting']:.4f}\n")
            
            # Add frame info overlay
            info_bg_height = 90
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, info_bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            info_text = f"Frame: {frame_idx}/{total_frames} | Humans: {len(human_results.boxes)} | Poses: {len(frame_predictions)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show shooting pose statistics
            shooting_percentage = (shooting_pose_count / total_detections * 100) if total_detections > 0 else 0
            stats_text = f"Shooting Poses: {shooting_pose_count}/{total_detections} ({shooting_percentage:.1f}%)"
            cv2.putText(
                annotated_frame,
                stats_text,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),  # Yellow
                2
            )
            
            # Write frame
            out.write(annotated_frame)
            
            # Show frame if requested
            if show:
                cv2.imshow('Shooting Pose Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            # Progress update
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames) | "
                      f"Shooting poses: {shooting_pose_count}/{total_detections}", end='\r')
            
            frame_idx += 1
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    
    # Export to CSV if requested
    if output_csv and all_prediction_data:
        csv_file = output_path / f"{video_name}_predictions_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            # Create CSV header
            fieldnames = ['frame_number', 'timestamp_seconds', 'timestamp_formatted', 
                         'human_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 
                         'human_conf', 'prediction', 'prob_non_shooting', 'prob_shooting']
            fieldnames.extend(feature_names)
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write each prediction entry
            for entry in all_prediction_data:
                timestamp_sec = entry['timestamp_seconds']
                
                # Format timestamp as HH:MM:SS.mmm
                hours = int(timestamp_sec // 3600)
                minutes = int((timestamp_sec % 3600) // 60)
                seconds = timestamp_sec % 60
                time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                
                entry['timestamp_formatted'] = time_formatted
                writer.writerow(entry)
        
        print(f"CSV data saved to: {csv_file}")
    
    # Export to JSON if requested
    if output_json and all_prediction_data:
        json_file = output_path / f"{video_name}_predictions_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'video_info': {
                    'source': str(video_path),
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'total_frames': total_frames,
                    'processed_frames': frame_idx
                },
                'model_info': {
                    'xgboost_model': str(xgboost_model_path),
                    'prediction_threshold': prediction_threshold
                },
                'statistics': {
                    'total_detections': total_detections,
                    'shooting_poses': shooting_pose_count,
                    'non_shooting_poses': total_detections - shooting_pose_count,
                    'shooting_percentage': (shooting_pose_count / total_detections * 100) if total_detections > 0 else 0
                },
                'feature_names': feature_names,
                'predictions': all_prediction_data
            }, f, indent=2)
        
        print(f"JSON data saved to: {json_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Processed frames: {frame_idx}")
    print(f"Total human detections: {total_detections}")
    print(f"Shooting poses detected: {shooting_pose_count} ({shooting_percentage:.1f}%)")
    print(f"Non-shooting poses: {total_detections - shooting_pose_count} ({100 - shooting_percentage:.1f}%)")
    print(f"\nOutput files:")
    print(f"  - Video: {output_video_path}")
    if output_csv and all_prediction_data:
        print(f"  - CSV: {csv_file}")
    if output_json and all_prediction_data:
        print(f"  - JSON: {json_file}")
    if save_txt:
        print(f"  - Text files: {txt_output_dir}")
    print(f"{'='*60}")


# ==================== COMMAND LINE INTERFACE ====================

def main():
    parser = argparse.ArgumentParser(
        description="Shooting Pose Detection - Human Detection + Pose Estimation + XGBoost Classification"
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--human-detector',
        type=str,
        default='/home/pavan_kadambala_invisible_email/Dev/cows/models/detector/models/HumanDetector-v2.pt',
        help='Path to human detector model (default: HumanDetector-v2.pt)'
    )
    
    parser.add_argument(
        '--pose-model',
        type=str,
        default='/home/pavan_kadambala_invisible_email/Dev/cows/models/detector/models/yolo11n-pose.pt',
        help='Path to pose model (default: yolo11n-pose.pt)'
    )
    
    parser.add_argument(
        '--xgboost-model',
        type=str,
        default='/home/pavan_kadambala_invisible_email/Dev/cows/models/detector/shooting_pose_xgboost_model.json',
        help='Path to XGBoost shooting pose classifier (default: shooting_pose_xgboost_model.json)'
    )
    
    parser.add_argument(
        '--feature-names',
        type=str,
        default='/home/pavan_kadambala_invisible_email/Dev/cows/models/detector/feature_names.txt',
        help='Path to feature names file (optional, for verification)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--conf-human',
        type=float,
        default=0.5,
        help='Confidence threshold for human detection (default: 0.5)'
    )
    
    parser.add_argument(
        '--conf-pose',
        type=float,
        default=0.25,
        help='Confidence threshold for pose estimation (default: 0.25)'
    )
    
    parser.add_argument(
        '--prediction-threshold',
        type=float,
        default=0.5,
        help='Threshold for binary classification (default: 0.5)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run on (default: cuda:0, use "cpu" for CPU)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show real-time results'
    )
    
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save predictions to text files'
    )
    
    parser.add_argument(
        '--output-csv',
        action='store_true',
        default=True,
        help='Save predictions to a consolidated CSV file (default: True)'
    )
    
    parser.add_argument(
        '--no-output-csv',
        dest='output_csv',
        action='store_false',
        help='Do not save predictions to CSV file'
    )
    
    parser.add_argument(
        '--output-json',
        action='store_true',
        help='Save predictions to a consolidated JSON file'
    )
    
    parser.add_argument(
        '--model-format',
        type=str,
        choices=['json', 'pickle'],
        default='json',
        help='XGBoost model format (default: json)'
    )
    
    args = parser.parse_args()
    
    # Run shooting pose detection
    run_shooting_pose_detection(
        video_path=args.source,
        human_detector_path=args.human_detector,
        pose_model_path=args.pose_model,
        xgboost_model_path=args.xgboost_model,
        output_dir=args.output_dir,
        conf_human=args.conf_human,
        conf_pose=args.conf_pose,
        prediction_threshold=args.prediction_threshold,
        device=args.device,
        show=args.show,
        save_txt=args.save_txt,
        output_csv=args.output_csv,
        output_json=args.output_json,
        model_format=args.model_format,
        feature_names_path=args.feature_names
    )


if __name__ == "__main__":
    main()


