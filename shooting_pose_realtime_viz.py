"""
Human Detection + Pose Estimation Pipeline with Real-Time Shooting Pose Metrics Visualization
This script runs human detection and pose estimation with side-by-side video and real-time graphs
showing key shooting pose metrics.

Metrics visualized:
1. Left wrist horizontal offset from shoulder midpoint (lateral positioning)
2. Right wrist horizontal offset from shoulder midpoint (lateral positioning)
3. Left wrist vertical offset from shoulder level (hand height)
4. Right wrist vertical offset from shoulder level (hand height)
5. Elbow angle (most extended arm, typically the supporting arm in shooting)
6. Inter-wrist distance normalized by shoulder width
7. Relative height of front wrist to eye level (normalized by torso height)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
from datetime import datetime
import csv
import json
from collections import deque
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


# ==================== GEOMETRIC UTILITY FUNCTIONS ====================

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(p2 - p1)


def calculate_angle_from_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate angle from vertical axis for vector p1->p2.
    
    Returns:
        Angle in degrees from vertical (0° = perfectly vertical, 90° = horizontal)
        Positive = leaning right, Negative = leaning left
    """
    vector = p2 - p1
    if np.linalg.norm(vector) < 1e-6:
        return np.nan
    
    # Calculate angle from vertical, preserving sign for direction
    angle_rad = np.arctan2(vector[0], vector[1])  # x, y (y increases downward)
    return np.degrees(angle_rad)


def calculate_tilt_from_horizontal(left_point: np.ndarray, right_point: np.ndarray) -> float:
    """
    Calculate tilt angle from horizontal between two points.
    Positive = right side higher, Negative = left side higher.
    
    Returns:
        Angle in degrees from horizontal
    """
    if np.linalg.norm(right_point - left_point) < 1e-6:
        return np.nan
    
    dy = right_point[1] - left_point[1]  # y increases downward
    dx = right_point[0] - left_point[0]
    
    angle = np.arctan2(-dy, dx)  # Negative dy because y increases downward
    return np.degrees(angle)


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at point p2 formed by p1-p2-p3.
    
    Args:
        p1, p2, p3: Points as [x, y] numpy arrays
    
    Returns:
        Angle in degrees (0-180)
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Handle zero vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return np.nan
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def get_midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Calculate midpoint between two points."""
    return (p1 + p2) / 2.0


# ==================== SHOOTING METRICS CALCULATOR ====================

class ShootingMetricsCalculator:
    """
    Calculates specific shooting pose metrics for real-time visualization.
    """
    
    # COCO keypoint indices
    KEYPOINT_INDICES = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the calculator.
        
        Args:
            confidence_threshold: Minimum confidence for using a keypoint
        """
        self.confidence_threshold = confidence_threshold
    
    def _get_keypoint(self, keypoints: np.ndarray, keypoint_conf: np.ndarray, 
                      name: str) -> Optional[np.ndarray]:
        """Get a keypoint if it exists and has sufficient confidence."""
        idx = self.KEYPOINT_INDICES[name]
        if keypoint_conf[idx] >= self.confidence_threshold:
            return keypoints[idx]
        return None
    
    def calculate_metrics(self, keypoints: np.ndarray, keypoint_conf: np.ndarray) -> Dict:
        """
        Calculate all shooting pose metrics for visualization.
        
        Args:
            keypoints: Array of shape (17, 2) with x, y coordinates
            keypoint_conf: Array of shape (17,) with confidence scores
        
        Returns:
            Dictionary with calculated metrics
        """
        # Extract keypoints
        kp = {}
        for name in self.KEYPOINT_INDICES.keys():
            kp[name] = self._get_keypoint(keypoints, keypoint_conf, name)
        
        metrics = {
            'left_wrist_horizontal_offset': None,
            'right_wrist_horizontal_offset': None,
            'left_wrist_vertical_offset': None,
            'right_wrist_vertical_offset': None,
            'elbow_angle': None,
            'normalized_wrist_distance': None,
            'wrist_eye_height_ratio': None
        }
        
        # Calculate shoulder midpoint for reference
        shoulder_midpoint = None
        shoulder_y = None
        if kp['left_shoulder'] is not None and kp['right_shoulder'] is not None:
            shoulder_midpoint = get_midpoint(kp['left_shoulder'], kp['right_shoulder'])
            shoulder_y = shoulder_midpoint[1]
        
        # 1. Left wrist horizontal offset from shoulder midpoint
        # Positive = left of center, Negative = right of center
        if shoulder_midpoint is not None and kp['left_wrist'] is not None:
            metrics['left_wrist_horizontal_offset'] = kp['left_wrist'][0] - shoulder_midpoint[0]
        
        # 2. Right wrist horizontal offset from shoulder midpoint
        # Positive = right of center, Negative = left of center
        if shoulder_midpoint is not None and kp['right_wrist'] is not None:
            metrics['right_wrist_horizontal_offset'] = kp['right_wrist'][0] - shoulder_midpoint[0]
        
        # 3. Left wrist vertical offset from shoulder level
        # Positive = below shoulder level, Negative = above shoulder level
        if shoulder_y is not None and kp['left_wrist'] is not None:
            metrics['left_wrist_vertical_offset'] = kp['left_wrist'][1] - shoulder_y
        
        # 4. Right wrist vertical offset from shoulder level
        # Positive = below shoulder level, Negative = above shoulder level
        if shoulder_y is not None and kp['right_wrist'] is not None:
            metrics['right_wrist_vertical_offset'] = kp['right_wrist'][1] - shoulder_y
        
        # 5. Elbow angle (use the more extended arm - typically the supporting arm)
        elbow_angles = []
        
        # Left elbow angle
        if (kp['left_shoulder'] is not None and kp['left_elbow'] is not None and 
            kp['left_wrist'] is not None):
            left_angle = calculate_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])
            if not np.isnan(left_angle):
                elbow_angles.append(left_angle)
        
        # Right elbow angle
        if (kp['right_shoulder'] is not None and kp['right_elbow'] is not None and 
            kp['right_wrist'] is not None):
            right_angle = calculate_angle(kp['right_shoulder'], kp['right_elbow'], kp['right_wrist'])
            if not np.isnan(right_angle):
                elbow_angles.append(right_angle)
        
        # Use the larger angle (more extended arm, typically the supporting arm in shooting)
        if elbow_angles:
            metrics['elbow_angle'] = max(elbow_angles)
        
        # 6. Inter-wrist distance normalized by shoulder width
        if (kp['left_wrist'] is not None and kp['right_wrist'] is not None and
            kp['left_shoulder'] is not None and kp['right_shoulder'] is not None):
            wrist_distance = calculate_distance(kp['left_wrist'], kp['right_wrist'])
            shoulder_width = calculate_distance(kp['left_shoulder'], kp['right_shoulder'])
            if shoulder_width > 1e-6:
                metrics['normalized_wrist_distance'] = wrist_distance / shoulder_width
        
        # 7. Relative height of front wrist to eye level (normalized by torso height)
        # This indicates if the hand is at eye level for aiming
        if kp['nose'] is not None or (kp['left_eye'] is not None and kp['right_eye'] is not None):
            # Get eye/head reference point
            if kp['left_eye'] is not None and kp['right_eye'] is not None:
                eye_y = get_midpoint(kp['left_eye'], kp['right_eye'])[1]
            elif kp['nose'] is not None:
                eye_y = kp['nose'][1]
            else:
                eye_y = None
            
            if eye_y is not None:
                # Find the closest wrist (front wrist)
                closest_wrist = None
                closest_dist = float('inf')
                
                if kp['nose'] is not None:
                    if kp['left_wrist'] is not None:
                        dist = calculate_distance(kp['nose'], kp['left_wrist'])
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_wrist = kp['left_wrist']
                    if kp['right_wrist'] is not None:
                        dist = calculate_distance(kp['nose'], kp['right_wrist'])
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_wrist = kp['right_wrist']
                
                if closest_wrist is not None:
                    # Calculate vertical distance (negative = wrist above eye, positive = below)
                    # In image coords, y increases downward
                    vertical_diff = closest_wrist[1] - eye_y
                    
                    # Normalize by torso height for scale independence
                    if (kp['left_shoulder'] is not None and kp['right_shoulder'] is not None and
                        kp['left_hip'] is not None and kp['right_hip'] is not None):
                        shoulder_y = get_midpoint(kp['left_shoulder'], kp['right_shoulder'])[1]
                        hip_y = get_midpoint(kp['left_hip'], kp['right_hip'])[1]
                        torso_height = abs(hip_y - shoulder_y)
                        
                        if torso_height > 1e-6:
                            # Normalized: negative = wrist above eye (good for aiming), positive = below
                            metrics['wrist_eye_height_ratio'] = vertical_diff / torso_height
        
        return metrics


# ==================== REAL-TIME GRAPH VISUALIZER ====================

class RealtimeGraphVisualizer:
    """
    Creates real-time graphs showing shooting pose metrics over time.
    """
    
    def __init__(self, graph_width: int = 800, graph_height: int = 1080, 
                 buffer_size: int = 300, fps: int = 30):
        """
        Initialize the visualizer.
        
        Args:
            graph_width: Width of graph area in pixels
            graph_height: Height of graph area in pixels
            buffer_size: Number of frames to show in graphs
            fps: Video frame rate for time calculations
        """
        self.graph_width = graph_width
        self.graph_height = graph_height
        self.buffer_size = buffer_size
        self.fps = fps
        
        # Data buffers for each metric
        self.frame_numbers = deque(maxlen=buffer_size)
        self.left_wrist_h_offset = deque(maxlen=buffer_size)
        self.right_wrist_h_offset = deque(maxlen=buffer_size)
        self.left_wrist_v_offset = deque(maxlen=buffer_size)
        self.right_wrist_v_offset = deque(maxlen=buffer_size)
        self.elbow_angle = deque(maxlen=buffer_size)
        self.norm_wrist_dist = deque(maxlen=buffer_size)
        self.wrist_eye_height = deque(maxlen=buffer_size)
        
        # Setup matplotlib figure
        self.fig = Figure(figsize=(graph_width/100, graph_height/100), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        
        # Create subplots (7 metrics stacked vertically)
        self.axes = []
        for i in range(7):
            ax = self.fig.add_subplot(7, 1, i+1)
            self.axes.append(ax)
        
        # Tight layout to maximize space
        self.fig.tight_layout(pad=2.0)
        
        # Metric configurations (name, color, ideal_range, units)
        self.metric_configs = [
            {
                'name': 'Left Wrist Horizontal Offset',
                'color': '#2E86AB',
                'ylabel': 'Offset (px)',
                'ideal_min': None,
                'ideal_max': None
            },
            {
                'name': 'Right Wrist Horizontal Offset',
                'color': '#1B5E7E',
                'ylabel': 'Offset (px)',
                'ideal_min': None,
                'ideal_max': None
            },
            {
                'name': 'Left Wrist Vertical Offset',
                'color': '#A23B72',
                'ylabel': 'Offset (px)',
                'ideal_min': None,
                'ideal_max': None
            },
            {
                'name': 'Right Wrist Vertical Offset',
                'color': '#E85D75',
                'ylabel': 'Offset (px)',
                'ideal_min': None,
                'ideal_max': None
            },
            {
                'name': 'Elbow Angle',
                'color': '#F18F01',
                'ylabel': 'Angle (deg)',
                'ideal_min': 90,
                'ideal_max': 180
            },
            {
                'name': 'Normalized Wrist Distance',
                'color': '#C73E1D',
                'ylabel': 'Ratio',
                'ideal_min': None,
                'ideal_max': None
            },
            {
                'name': 'Wrist-Eye Height Ratio',
                'color': '#9D4EDD',
                'ylabel': 'Normalized Height',
                'ideal_min': -0.3,
                'ideal_max': 0.1
            }
        ]
    
    def update_data(self, frame_number: int, metrics: Dict):
        """
        Update data buffers with new metrics.
        
        Args:
            frame_number: Current frame number
            metrics: Dictionary of calculated metrics
        """
        self.frame_numbers.append(frame_number)
        self.left_wrist_h_offset.append(metrics.get('left_wrist_horizontal_offset'))
        self.right_wrist_h_offset.append(metrics.get('right_wrist_horizontal_offset'))
        self.left_wrist_v_offset.append(metrics.get('left_wrist_vertical_offset'))
        self.right_wrist_v_offset.append(metrics.get('right_wrist_vertical_offset'))
        self.elbow_angle.append(metrics.get('elbow_angle'))
        self.norm_wrist_dist.append(metrics.get('normalized_wrist_distance'))
        self.wrist_eye_height.append(metrics.get('wrist_eye_height_ratio'))
    
    def render_graphs(self, current_frame: int) -> np.ndarray:
        """
        Render all graphs to a numpy array.
        
        Args:
            current_frame: Current frame number for marking
        
        Returns:
            RGB numpy array of the rendered graphs
        """
        # Data arrays for plotting
        data_buffers = [
            self.left_wrist_h_offset,
            self.right_wrist_h_offset,
            self.left_wrist_v_offset,
            self.right_wrist_v_offset,
            self.elbow_angle,
            self.norm_wrist_dist,
            self.wrist_eye_height
        ]
        
        frame_nums = list(self.frame_numbers)
        
        # Clear all axes
        for ax in self.axes:
            ax.clear()
        
        # Plot each metric
        for idx, (ax, data_buffer, config) in enumerate(zip(self.axes, data_buffers, self.metric_configs)):
            # Convert to lists and filter out None values
            data_list = list(data_buffer)
            
            # Prepare data for plotting (keep indices aligned with frames)
            plot_frames = []
            plot_values = []
            for i, val in enumerate(data_list):
                if val is not None and not np.isnan(val):
                    plot_frames.append(frame_nums[i] if i < len(frame_nums) else i)
                    plot_values.append(val)
            
            # Plot the data
            if plot_frames and plot_values:
                ax.plot(plot_frames, plot_values, 
                       color=config['color'], linewidth=2, label=config['name'])
                
                # Mark current frame
                if plot_frames and current_frame >= plot_frames[0]:
                    ax.axvline(x=current_frame, color='red', linestyle='--', 
                             linewidth=1, alpha=0.7, label='Current')
                
                # Add ideal range if specified
                if config['ideal_min'] is not None and config['ideal_max'] is not None:
                    ax.axhspan(config['ideal_min'], config['ideal_max'], 
                             alpha=0.2, color='green', label='Ideal Range')
            
            # Styling
            ax.set_xlabel('Frame', fontsize=8)
            ax.set_ylabel(config['ylabel'], fontsize=8)
            ax.set_title(config['name'], fontsize=10, fontweight='bold', pad=5)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=7)
            
            # Set x-axis limits to show buffer window
            if frame_nums:
                min_frame = max(0, current_frame - self.buffer_size)
                max_frame = max(current_frame + 10, self.buffer_size)
                ax.set_xlim(min_frame, max_frame)
            
            # Add legend if we have data
            if plot_frames and plot_values:
                ax.legend(fontsize=6, loc='upper right')
        
        # Render to numpy array
        self.canvas.draw()
        buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        
        # Convert RGBA to RGB
        rgb_buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
        
        return rgb_buf


# ==================== MAIN INFERENCE FUNCTION ====================

def run_shooting_pose_realtime_viz(
    video_path,
    human_detector_path,
    pose_model_path,
    output_dir="output",
    conf_human=0.5,
    conf_pose=0.25,
    device="cuda:0",
    show=False,
    save_txt=False,
    output_csv=True,
    output_json=False,
    save_plot=False,
    graph_buffer_size=300
):
    """
    Run human detection + pose estimation with real-time metric visualization.
    
    Args:
        video_path: Path to input video
        human_detector_path: Path to human detector model
        pose_model_path: Path to pose estimation model
        output_dir: Output directory for results
        conf_human: Confidence threshold for human detection
        conf_pose: Confidence threshold for pose estimation
        device: Device to run on ('cuda:0' or 'cpu')
        show: Show real-time results
        save_txt: Save metrics to text files
        output_csv: Save metrics to a consolidated CSV file (default: True)
        output_json: Save metrics to a consolidated JSON file (default: False)
        save_plot: Save final metrics plot as PNG image (default: False)
        graph_buffer_size: Number of frames to show in graphs
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
    
    # Initialize metrics calculator
    metrics_calculator = ShootingMetricsCalculator(confidence_threshold=0.5)
    
    # Initialize graph visualizer
    graph_width = 800
    graph_visualizer = RealtimeGraphVisualizer(
        graph_width=graph_width,
        graph_height=height,
        buffer_size=graph_buffer_size,
        fps=fps
    )
    
    # Setup video writer (video + graphs side by side)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem
    output_video_path = output_path / f"{video_name}_realtime_viz_{timestamp}.mp4"
    
    combined_width = width + graph_width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (combined_width, height))
    
    # Setup text output if needed
    if save_txt:
        txt_output_dir = output_path / f"{video_name}_metrics_{timestamp}"
        txt_output_dir.mkdir(exist_ok=True)
    
    # Collect all metrics for CSV/JSON export
    all_metrics_data = []
    
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
    
    print(f"\nProcessing video with real-time visualization...")
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
            
            # Default metrics (None if no detection)
            frame_metrics = {
                'left_wrist_horizontal_offset': None,
                'right_wrist_horizontal_offset': None,
                'left_wrist_vertical_offset': None,
                'right_wrist_vertical_offset': None,
                'elbow_angle': None,
                'normalized_wrist_distance': None,
                'wrist_eye_height_ratio': None
            }
            
            # Step 2: For each detected human, run pose estimation
            # (We'll focus on the first/most confident detection for visualization)
            if len(human_results.boxes) > 0:
                # Get first (most confident) detection
                box = human_results.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Draw bounding box for human detection
                if x2 - x1 >= 20 and y2 - y1 >= 20:
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
                    
                    if cropped_human.size > 0:
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
                            
                            # Calculate shooting metrics
                            frame_metrics = metrics_calculator.calculate_metrics(keypoints, keypoint_conf)
                            
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
                            
                            # Display metrics as text overlay on video
                            y_offset = 60
                            text_color = (255, 255, 255)
                            bg_color = (0, 0, 0)
                            
                            metric_texts = [
                                f"L Wrist H-Offset: {frame_metrics['left_wrist_horizontal_offset']:.1f}px" if frame_metrics['left_wrist_horizontal_offset'] is not None else "L Wrist H-Offset: N/A",
                                f"R Wrist H-Offset: {frame_metrics['right_wrist_horizontal_offset']:.1f}px" if frame_metrics['right_wrist_horizontal_offset'] is not None else "R Wrist H-Offset: N/A",
                                f"L Wrist V-Offset: {frame_metrics['left_wrist_vertical_offset']:.1f}px" if frame_metrics['left_wrist_vertical_offset'] is not None else "L Wrist V-Offset: N/A",
                                f"R Wrist V-Offset: {frame_metrics['right_wrist_vertical_offset']:.1f}px" if frame_metrics['right_wrist_vertical_offset'] is not None else "R Wrist V-Offset: N/A",
                                f"Elbow Angle: {frame_metrics['elbow_angle']:.1f}deg" if frame_metrics['elbow_angle'] is not None else "Elbow Angle: N/A",
                                f"Norm Wrist: {frame_metrics['normalized_wrist_distance']:.2f}" if frame_metrics['normalized_wrist_distance'] is not None else "Norm Wrist: N/A",
                                f"Wrist-Eye Ht: {frame_metrics['wrist_eye_height_ratio']:.2f}" if frame_metrics['wrist_eye_height_ratio'] is not None else "Wrist-Eye Ht: N/A"
                            ]
                            
                            for text in metric_texts:
                                # Draw background rectangle
                                (text_width, text_height), _ = cv2.getTextSize(
                                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                                )
                                cv2.rectangle(
                                    annotated_frame,
                                    (5, y_offset - text_height - 2),
                                    (15 + text_width, y_offset + 2),
                                    bg_color,
                                    -1
                                )
                                # Draw text
                                cv2.putText(
                                    annotated_frame,
                                    text,
                                    (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    text_color,
                                    1
                                )
                                y_offset += 25
            
            # Update graph visualizer with current metrics
            graph_visualizer.update_data(frame_idx, frame_metrics)
            
            # Render graphs
            graph_image = graph_visualizer.render_graphs(frame_idx)
            
            # Resize graph if needed to match video height
            if graph_image.shape[0] != height:
                graph_image = cv2.resize(graph_image, (graph_width, height))
            
            # Add frame info to video
            info_text = f"Frame: {frame_idx}/{total_frames}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Combine video and graphs side by side
            combined_frame = np.hstack([annotated_frame, graph_image])
            
            # Write combined frame
            out.write(combined_frame)
            
            # Show frame if requested
            if show:
                try:
                    # Resize for display if too large
                    display_frame = combined_frame
                    if combined_width > 1920:
                        scale = 1920 / combined_width
                        display_width = int(combined_width * scale)
                        display_height = int(height * scale)
                        display_frame = cv2.resize(combined_frame, (display_width, display_height))
                    
                    cv2.imshow('Shooting Pose Real-Time Visualization', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
                except cv2.error as e:
                    if frame_idx == 0:  # Only print warning once
                        print("\nWarning: Cannot display video (OpenCV GUI not available)")
                        print("Continuing processing without display. Output will still be saved.")
                        show = False  # Disable show for remaining frames
            
            # Save to text file if requested
            if save_txt:
                txt_file = txt_output_dir / f"frame_{frame_idx:06d}.txt"
                with open(txt_file, 'w') as f:
                    f.write(f"Frame: {frame_idx}\n")
                    f.write(f"Timestamp: {frame_idx / fps:.3f}s\n\n")
                    f.write("Shooting Pose Metrics:\n")
                    for key, value in frame_metrics.items():
                        if value is not None:
                            f.write(f"  {key}: {value:.3f}\n")
                        else:
                            f.write(f"  {key}: N/A\n")
            
            # Collect for CSV/JSON export
            if output_csv or output_json:
                timestamp_seconds = frame_idx / fps if fps > 0 else 0
                
                data_entry = {
                    'frame_number': frame_idx,
                    'timestamp_seconds': timestamp_seconds,
                    **frame_metrics
                }
                
                all_metrics_data.append(data_entry)
            
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
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignore if GUI not available
    
    # Save final plot image with all metrics
    final_plot_path = output_path / f"{video_name}_metrics_plot_{timestamp}.png"
    
    if save_plot and all_metrics_data:
        print(f"\nGenerating final metrics plot...")
        
        # Create a new figure for the final plot
        fig_final = plt.figure(figsize=(16, 14))
        
        # Prepare data for all metrics
        all_frame_numbers = [entry['frame_number'] for entry in all_metrics_data]
        
        metric_data_lists = {
            'left_wrist_horizontal_offset': [],
            'right_wrist_horizontal_offset': [],
            'left_wrist_vertical_offset': [],
            'right_wrist_vertical_offset': [],
            'elbow_angle': [],
            'normalized_wrist_distance': [],
            'wrist_eye_height_ratio': []
        }
        
        # Extract data for each metric
        for entry in all_metrics_data:
            for key in metric_data_lists.keys():
                metric_data_lists[key].append(entry.get(key))
        
        # Metric configurations for final plot
        final_metric_configs = [
            ('Left Wrist Horizontal Offset', 'left_wrist_horizontal_offset', 'Offset (px)', '#2E86AB', None, None),
            ('Right Wrist Horizontal Offset', 'right_wrist_horizontal_offset', 'Offset (px)', '#1B5E7E', None, None),
            ('Left Wrist Vertical Offset', 'left_wrist_vertical_offset', 'Offset (px)', '#A23B72', None, None),
            ('Right Wrist Vertical Offset', 'right_wrist_vertical_offset', 'Offset (px)', '#E85D75', None, None),
            ('Elbow Angle', 'elbow_angle', 'Angle (degrees)', '#F18F01', 90, 180),
            ('Normalized Wrist Distance', 'normalized_wrist_distance', 'Ratio', '#C73E1D', None, None),
            ('Wrist-Eye Height Ratio', 'wrist_eye_height_ratio', 'Normalized Height', '#9D4EDD', -0.3, 0.1)
        ]
        
        # Plot each metric
        for idx, (name, key, ylabel, color, ideal_min, ideal_max) in enumerate(final_metric_configs):
            ax = fig_final.add_subplot(7, 1, idx + 1)
            
            # Get data and filter out None values
            frames = []
            values = []
            for frame_num, value in zip(all_frame_numbers, metric_data_lists[key]):
                if value is not None and not np.isnan(value):
                    frames.append(frame_num)
                    values.append(value)
            
            # Plot if we have data
            if frames and values:
                ax.plot(frames, values, color=color, linewidth=2, label=name)
                
                # Add ideal range if specified
                if ideal_min is not None and ideal_max is not None:
                    ax.axhspan(ideal_min, ideal_max, alpha=0.2, color='green', label='Ideal Range')
                    ax.legend(loc='upper right', fontsize=8)
            
            # Styling
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(name, fontsize=12, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=9)
            
            # Only show x-label on bottom plot
            if idx == len(final_metric_configs) - 1:
                ax.set_xlabel('Frame Number', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig_final)
        
        print(f"Final metrics plot saved to: {final_plot_path}")
    
    # Export to CSV if requested
    if output_csv and all_metrics_data:
        csv_file = output_path / f"{video_name}_metrics_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['frame_number', 'timestamp_seconds', 'timestamp_formatted',
                         'left_wrist_horizontal_offset', 'right_wrist_horizontal_offset',
                         'left_wrist_vertical_offset', 'right_wrist_vertical_offset',
                         'elbow_angle', 'normalized_wrist_distance', 'wrist_eye_height_ratio']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in all_metrics_data:
                timestamp_sec = entry['timestamp_seconds']
                
                # Format timestamp as HH:MM:SS.mmm
                hours = int(timestamp_sec // 3600)
                minutes = int((timestamp_sec % 3600) // 60)
                seconds = timestamp_sec % 60
                time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                
                row = {
                    'frame_number': entry['frame_number'],
                    'timestamp_seconds': timestamp_sec,
                    'timestamp_formatted': time_formatted,
                    'left_wrist_horizontal_offset': entry.get('left_wrist_horizontal_offset'),
                    'right_wrist_horizontal_offset': entry.get('right_wrist_horizontal_offset'),
                    'left_wrist_vertical_offset': entry.get('left_wrist_vertical_offset'),
                    'right_wrist_vertical_offset': entry.get('right_wrist_vertical_offset'),
                    'elbow_angle': entry.get('elbow_angle'),
                    'normalized_wrist_distance': entry.get('normalized_wrist_distance'),
                    'wrist_eye_height_ratio': entry.get('wrist_eye_height_ratio')
                }
                
                writer.writerow(row)
        
        print(f"\nCSV data saved to: {csv_file}")
    
    # Export to JSON if requested
    if output_json and all_metrics_data:
        json_file = output_path / f"{video_name}_metrics_{timestamp}.json"
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
                'metrics': [
                    'left_wrist_horizontal_offset',
                    'right_wrist_horizontal_offset',
                    'left_wrist_vertical_offset',
                    'right_wrist_vertical_offset',
                    'elbow_angle',
                    'normalized_wrist_distance',
                    'wrist_eye_height_ratio'
                ],
                'data': all_metrics_data
            }, f, indent=2)
        
        print(f"JSON data saved to: {json_file}")
    
    print(f"\n\nProcessing complete!")
    print(f"Processed {frame_idx} frames")
    print(f"\nOutput files:")
    print(f"  - Video: {output_video_path}")
    if output_csv and all_metrics_data:
        print(f"  - Metrics CSV: {csv_file}")
    if output_json and all_metrics_data:
        print(f"  - Metrics JSON: {json_file}")
    if save_plot and all_metrics_data and final_plot_path.exists():
        print(f"  - Metrics plot: {final_plot_path}")
    if save_txt:
        print(f"  - Text files: {txt_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Human Detection + Pose Estimation with Real-Time Shooting Metrics Visualization"
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
        help='Save metrics to text files per frame'
    )
    
    parser.add_argument(
        '--output-csv',
        action='store_true',
        default=True,
        help='Save metrics to a consolidated CSV file (default: True)'
    )
    
    parser.add_argument(
        '--no-output-csv',
        dest='output_csv',
        action='store_false',
        help='Do not save metrics to CSV file'
    )
    
    parser.add_argument(
        '--output-json',
        action='store_true',
        help='Save metrics to a consolidated JSON file'
    )
    
    parser.add_argument(
        '--save-plot',
        action='store_true',
        default=False,
        help='Save final metrics plot as PNG image (default: False)'
    )
    
    parser.add_argument(
        '--graph-buffer',
        type=int,
        default=300,
        help='Number of frames to show in graphs (default: 300, ~10 seconds at 30fps)'
    )
    
    args = parser.parse_args()
    
    # Run inference with real-time visualization
    run_shooting_pose_realtime_viz(
        video_path=args.source,
        human_detector_path=args.human_detector,
        pose_model_path=args.pose_model,
        output_dir=args.output_dir,
        conf_human=args.conf_human,
        conf_pose=args.conf_pose,
        device=args.device,
        show=args.show,
        save_txt=args.save_txt,
        output_csv=args.output_csv,
        output_json=args.output_json,
        save_plot=args.save_plot,
        graph_buffer_size=args.graph_buffer
    )


if __name__ == "__main__":
    main()

