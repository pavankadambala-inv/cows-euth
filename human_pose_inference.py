"""
Human Detection + Pose Estimation Pipeline
This script runs human detection first, then runs pose estimation on each detected human.
Enhanced with frame extraction and CSV export capabilities.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
from datetime import datetime
import csv
import json


def run_human_pose_inference(
    video_path,
    human_detector_path,
    pose_model_path,
    output_dir="output",
    conf_human=0.5,
    conf_pose=0.25,
    device="cuda:0",
    show=False,
    save_txt=False,
    save_frames=False,
    output_csv=False,
    output_json=False
):
    """
    Run human detection + pose estimation pipeline
    
    Args:
        video_path: Path to input video
        human_detector_path: Path to human detector model
        pose_model_path: Path to pose estimation model
        output_dir: Output directory for results
        conf_human: Confidence threshold for human detection
        conf_pose: Confidence threshold for pose estimation
        device: Device to run on ('cuda:0' or 'cpu')
        show: Show real-time results
        save_txt: Save pose keypoints to text files
        save_frames: Save individual frames as images
        output_csv: Save pose data to a consolidated CSV file
        output_json: Save pose data to a consolidated JSON file
    """
    
    # Load models
    print(f"Loading human detector from: {human_detector_path}")
    human_detector = YOLO(human_detector_path)
    
    print(f"Loading pose model from: {pose_model_path}")
    pose_model = YOLO(pose_model_path)
    
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
    output_video_path = output_path / f"{video_name}_human_pose_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Setup text output if needed
    if save_txt:
        txt_output_dir = output_path / f"{video_name}_poses_{timestamp}"
        txt_output_dir.mkdir(exist_ok=True)
    
    # Setup frames output if needed
    if save_frames:
        frames_output_dir = output_path / f"{video_name}_frames_{timestamp}"
        frames_output_dir.mkdir(exist_ok=True)
        print(f"Frames will be saved to: {frames_output_dir}")
    
    # Collect all pose data for CSV/JSON export
    all_pose_data = []
    
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
    
    print(f"\nProcessing video...")
    print(f"Output will be saved to: {output_video_path}")
    
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
            
            # Store all pose data for this frame
            frame_poses = []
            
            # Step 2: For each detected human, run pose estimation
            if len(human_results.boxes) > 0:
                for idx, box in enumerate(human_results.boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    
                    # Ensure coordinates are within frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # Skip if box is too small
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue
                    
                    # Draw bounding box for human detection
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"Human {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    
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
                    
                    # Process pose keypoints
                    if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
                        # Get the first (most confident) pose
                        keypoints = pose_results.keypoints[0].xy.cpu().numpy()[0]  # Shape: (17, 2)
                        keypoint_conf = pose_results.keypoints[0].conf.cpu().numpy()[0]  # Shape: (17,)
                        
                        # Transform keypoints back to original frame coordinates
                        keypoints[:, 0] += crop_x1
                        keypoints[:, 1] += crop_y1
                        
                        # Draw skeleton
                        for connection in skeleton_connections:
                            pt1_idx, pt2_idx = connection
                            if (keypoint_conf[pt1_idx] > 0.5 and 
                                keypoint_conf[pt2_idx] > 0.5):
                                pt1 = tuple(map(int, keypoints[pt1_idx]))
                                pt2 = tuple(map(int, keypoints[pt2_idx]))
                                cv2.line(annotated_frame, pt1, pt2, (255, 0, 0), 2)
                        
                        # Draw keypoints
                        for kp_idx, (kp, kp_conf) in enumerate(zip(keypoints, keypoint_conf)):
                            if kp_conf > 0.5:
                                x, y = map(int, kp)
                                cv2.circle(annotated_frame, (x, y), 4, (0, 0, 255), -1)
                        
                        # Store pose data
                        pose_data = {
                            'human_id': idx,
                            'bbox': [x1, y1, x2, y2],
                            'human_conf': conf,
                            'keypoints': keypoints.tolist(),
                            'keypoint_conf': keypoint_conf.tolist()
                        }
                        frame_poses.append(pose_data)
                        
                        # Collect for CSV/JSON export
                        if output_csv or output_json:
                            # Calculate timestamp for this frame
                            timestamp_seconds = frame_idx / fps if fps > 0 else 0
                            
                            all_pose_data.append({
                                'frame_number': frame_idx,
                                'timestamp_seconds': timestamp_seconds,
                                'human_id': idx,
                                'bbox': [x1, y1, x2, y2],
                                'human_conf': conf,
                                'keypoints': keypoints.tolist(),
                                'keypoint_conf': keypoint_conf.tolist()
                            })
            
            # Save original frame if requested
            if save_frames:
                frame_filename = frames_output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
            
            # Save pose data to text file if requested
            if save_txt and frame_poses:
                txt_file = txt_output_dir / f"frame_{frame_idx:06d}.txt"
                with open(txt_file, 'w') as f:
                    f.write(f"Frame: {frame_idx}\n")
                    for pose in frame_poses:
                        f.write(f"\nHuman {pose['human_id']}:\n")
                        f.write(f"  BBox: {pose['bbox']}\n")
                        f.write(f"  Confidence: {pose['human_conf']:.3f}\n")
                        f.write(f"  Keypoints:\n")
                        for kp_idx, (kp, kp_conf) in enumerate(zip(pose['keypoints'], pose['keypoint_conf'])):
                            f.write(f"    {kp_idx}: ({kp[0]:.1f}, {kp[1]:.1f}) conf={kp_conf:.3f}\n")
            
            # Add frame info
            info_text = f"Frame: {frame_idx}/{total_frames} | Humans: {len(human_results.boxes)} | Poses: {len(frame_poses)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Write frame
            out.write(annotated_frame)
            
            # Show frame if requested
            if show:
                cv2.imshow('Human Pose Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            # Progress update
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)", end='\r')
            
            frame_idx += 1
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()
    
    # Export to CSV if requested
    if output_csv and all_pose_data:
        csv_file = output_path / f"{video_name}_pose_data_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            # Create CSV header with timestamp columns
            fieldnames = ['frame_number', 'timestamp_seconds', 'timestamp_formatted', 'human_id', 
                         'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'human_conf']
            for kp_name in keypoint_names:
                fieldnames.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_conf'])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write each pose entry
            for pose_entry in all_pose_data:
                timestamp_sec = pose_entry['timestamp_seconds']
                
                # Format timestamp as HH:MM:SS.mmm
                hours = int(timestamp_sec // 3600)
                minutes = int((timestamp_sec % 3600) // 60)
                seconds = timestamp_sec % 60
                time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                
                row = {
                    'frame_number': pose_entry['frame_number'],
                    'timestamp_seconds': timestamp_sec,
                    'timestamp_formatted': time_formatted,
                    'human_id': pose_entry['human_id'],
                    'bbox_x1': pose_entry['bbox'][0],
                    'bbox_y1': pose_entry['bbox'][1],
                    'bbox_x2': pose_entry['bbox'][2],
                    'bbox_y2': pose_entry['bbox'][3],
                    'human_conf': pose_entry['human_conf']
                }
                
                # Add keypoint data
                for kp_idx, kp_name in enumerate(keypoint_names):
                    row[f'{kp_name}_x'] = pose_entry['keypoints'][kp_idx][0]
                    row[f'{kp_name}_y'] = pose_entry['keypoints'][kp_idx][1]
                    row[f'{kp_name}_conf'] = pose_entry['keypoint_conf'][kp_idx]
                
                writer.writerow(row)
        
        print(f"CSV data saved to: {csv_file}")
    
    # Export to JSON if requested
    if output_json and all_pose_data:
        json_file = output_path / f"{video_name}_pose_data_{timestamp}.json"
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
                'keypoint_names': keypoint_names,
                'pose_data': all_pose_data
            }, f, indent=2)
        
        print(f"JSON data saved to: {json_file}")
    
    print(f"\n\nProcessing complete!")
    print(f"Processed {frame_idx} frames")
    print(f"Output video saved to: {output_video_path}")
    if save_txt:
        print(f"Pose data saved to: {txt_output_dir}")
    if save_frames:
        print(f"Frames saved to: {frames_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Human Detection + Pose Estimation Pipeline")
    
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
        help='Save pose keypoints to text files'
    )
    
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save individual frames as images'
    )
    
    parser.add_argument(
        '--output-csv',
        action='store_true',
        help='Save pose data to a consolidated CSV file'
    )
    
    parser.add_argument(
        '--output-json',
        action='store_true',
        help='Save pose data to a consolidated JSON file'
    )
    
    args = parser.parse_args()
    
    # Run inference
    run_human_pose_inference(
        video_path=args.source,
        human_detector_path=args.human_detector,
        pose_model_path=args.pose_model,
        output_dir=args.output_dir,
        conf_human=args.conf_human,
        conf_pose=args.conf_pose,
        device=args.device,
        show=args.show,
        save_txt=args.save_txt,
        save_frames=args.save_frames,
        output_csv=args.output_csv,
        output_json=args.output_json
    )


if __name__ == "__main__":
    main()
