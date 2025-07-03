#!/usr/bin/env python3
"""
Blink Rate Monitoring using YOLOv8 and MediaPipe Face Mesh
Real-time blink detection and BPM calculation with live visualization
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from datetime import datetime
from datetime import datetime, timedelta

class BlinkDetector:
    def __init__(self):
        # Initialize models
        print("Loading YOLOv8n model...")
        self.yolo_model = YOLO('yolov8n.pt')  # Will download if not present
        
        print("Initializing MediaPipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Disable for better performance
            min_detection_confidence=0.8,  # Higher confidence = fewer false positives
            min_tracking_confidence=0.7    # Higher tracking confidence
        )
        
        # Eye landmark indices (MediaPipe 468-point model)
        # Standard EAR calculation uses 6 points per eye
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]   # Left eye (viewer's right)
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]   # Right eye (viewer's left)
        
        # Blink detection parameters
        self.EAR_THRESHOLD = 0.23  # Adjusted for single-frame detection
        self.BLINK_CONSECUTIVE_FRAMES = 1  # Minimum frames for blink detection
        
        # State tracking
        self.blink_counter = 0
        self.blink_timestamps = deque(maxlen=100)  # Store last 100 blinks
        self.ear_history = deque(maxlen=300)  # Store EAR history for smoothing
        self.eye_closed_counter = 0
        self.last_face_region = None
        self.face_detection_failures = 0
        
        # Performance optimization
        self.yolo_skip_frames = 15  # Increase to 15 for better performance
        self.frame_count = 0
        self.last_valid_face_coords = None
        self.face_coords_valid_frames = 0  # Track how long face coords are valid
        
        # Static display window settings
        self.DISPLAY_WIDTH = 480   # Fixed display width
        self.DISPLAY_HEIGHT = 360  # Fixed display height
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Live plotting setup (disabled by default for performance)
        self.plotting_enabled = False  # Set to True to enable live graphs
        self.debug_mode = False  # Minimize debug output by default
        
        # New 5-minute plotting system
        self.plot_interval = 5.0  # 5 seconds per data point
        self.plot_window_duration = 300.0  # 5 minutes total window
        self.plot_start_time = None
        self.current_interval_start = None
        self.current_interval_blinks = []
        self.current_interval_ears = []
        
        # Data storage for 5-minute windows
        self.window_times = []  # Time points for current 5-minute window
        self.window_bpm = []    # BPM values for current 5-minute window
        self.window_ear = []    # EAR values for current 5-minute window
        
        # Plot counter for saving files
        self.plot_save_counter = 0
        
        if self.plotting_enabled:
            self.setup_live_plot()
        else:
            print("Live plotting disabled for better performance. Press 'p' to enable.")
            self.fig = None
            self.ax1 = None
            self.ax2 = None
        
    def setup_live_plot(self):
        """Initialize live plotting for BPM and EAR visualization"""
        try:
            # Use a more stable backend for matplotlib
            plt.switch_backend('TkAgg')  # More stable for interactive plots
            
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # BPM plot
            self.ax1.set_title('Blinks Per Minute (BPM) - Live')
            self.ax1.set_ylabel('BPM')
            self.ax1.set_ylim(0, 30)
            self.ax1.grid(True, alpha=0.3)
            
            # EAR plot
            self.ax2.set_title('Eye Aspect Ratio (EAR) - Live')
            self.ax2.set_ylabel('EAR')
            self.ax2.set_xlabel('Time (seconds)')
            self.ax2.set_ylim(0.1, 0.4)
            self.ax2.axhline(y=self.EAR_THRESHOLD, color='r', linestyle='--', label='Threshold')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()
            
            plt.tight_layout()
            plt.ion()  # Interactive mode
            
            # Set window properties to make it more stable
            if hasattr(self.fig.canvas.manager, 'window'):
                try:
                    self.fig.canvas.manager.window.wm_title("Blink Detector - Live Graphs")
                except:
                    pass  # Ignore if title setting fails
            
            plt.show(block=False)  # Non-blocking show
            
            # Small delay to ensure window is properly created
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Warning: Could not initialize live plotting: {e}")
            print("Continuing without live graphs...")
            self.fig = None
            self.ax1 = None
            self.ax2 = None
        
    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio for given eye landmarks"""
        try:
            # Get eye landmarks
            eye_points = []
            for idx in eye_indices:
                point = landmarks[idx]
                eye_points.append([point.x, point.y])
            
            eye_points = np.array(eye_points)
            
            # Debug: Print eye points to understand the issue
            if len(eye_points) != 6:
                print(f"WARNING: Expected 6 eye points, got {len(eye_points)}")
                return 0.0
            
            # Calculate EAR: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
            # Vertical distances
            vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
            vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
            
            # Horizontal distance
            horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
            
            # Debug: Print distance calculations
            if horizontal == 0:
                print(f"WARNING: Horizontal distance is 0")
                return 0.0
            
            # EAR calculation
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            
            # Minimal debug output only in debug mode
            if self.debug_mode and hasattr(self, 'debug_counter'):
                self.debug_counter += 1
                if self.debug_counter % 300 == 0:  # Much less frequent
                    print(f"DEBUG EAR: {ear:.4f} (V1:{vertical_1:.4f}, V2:{vertical_2:.4f}, H:{horizontal:.4f})")
            elif not hasattr(self, 'debug_counter'):
                self.debug_counter = 0
            
            return ear
                
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.0
    
    def create_fixed_display_frame(self, face_region, full_frame):
        """Create a fixed-size display frame, centering the face region"""
        # Create a black canvas of fixed size
        display_frame = np.zeros((self.DISPLAY_HEIGHT, self.DISPLAY_WIDTH, 3), dtype=np.uint8)
        
        if face_region is not None:
            # Resize face region to fit within our display while maintaining aspect ratio
            face_h, face_w = face_region.shape[:2]
            
            # Calculate scaling to fit within display bounds with some padding
            scale_w = (self.DISPLAY_WIDTH - 40) / face_w  # 20px padding each side
            scale_h = (self.DISPLAY_HEIGHT - 60) / face_h  # 30px padding top/bottom for text
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only downscale
            
            # Resize face region
            new_w = int(face_w * scale)
            new_h = int(face_h * scale)
            resized_face = cv2.resize(face_region, (new_w, new_h))
            
            # Center the face in the display frame
            start_x = (self.DISPLAY_WIDTH - new_w) // 2
            start_y = (self.DISPLAY_HEIGHT - new_h) // 2
            
            # Place the face region in the center
            display_frame[start_y:start_y+new_h, start_x:start_x+new_w] = resized_face
            
            # Add face detection indicator (green border)
            cv2.rectangle(display_frame, (start_x-2, start_y-2), 
                         (start_x+new_w+2, start_y+new_h+2), (0, 255, 0), 2)
            
            # Add "FACE DETECTED" text at top
            cv2.putText(display_frame, "FACE DETECTED", (5, 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return display_frame, (start_x, start_y, new_w, new_h)
        else:
            # No face - show a centered portion of the full frame
            full_h, full_w = full_frame.shape[:2]
            
            # Calculate crop area (center of full frame)
            crop_w = min(full_w, int(self.DISPLAY_WIDTH * 1.5))
            crop_h = min(full_h, int(self.DISPLAY_HEIGHT * 1.5))
            
            center_x = full_w // 2
            center_y = full_h // 2
            
            crop_x1 = max(0, center_x - crop_w // 2)
            crop_y1 = max(0, center_y - crop_h // 2)
            crop_x2 = min(full_w, crop_x1 + crop_w)
            crop_y2 = min(full_h, crop_y1 + crop_h)
            
            # Crop and resize to fit display
            cropped = full_frame[crop_y1:crop_y2, crop_x1:crop_x2]
            display_frame = cv2.resize(cropped, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
            
            # Add "NO FACE DETECTED" text
            cv2.putText(display_frame, "NO FACE DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            return display_frame, None
    
    def detect_face_yolo(self, frame):
        """Detect face using YOLOv8 and return cropped face region"""
        try:
            results = self.yolo_model(frame, classes=[0], verbose=False)  # Class 0 = person
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get the largest detection (closest person)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                # Find the box with highest confidence
                best_idx = np.argmax(confidences)
                x1, y1, x2, y2 = boxes[best_idx].astype(int)
                
                # Expand the bounding box slightly for better face mesh detection
                h, w = frame.shape[:2]
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                face_region = frame[y1:y2, x1:x2]
                self.last_face_region = (x1, y1, x2, y2)
                self.face_detection_failures = 0
                
                return face_region, (x1, y1, x2, y2)
            else:
                self.face_detection_failures += 1
                return None, None
                
        except Exception as e:
            print(f"Error in face detection: {e}")
            self.face_detection_failures += 1
            return None, None
    
    def calculate_bpm(self):
        """Calculate blinks per minute using last 60 seconds of data"""
        if len(self.blink_timestamps) < 2:
            return 0.0
            
        current_time = time.time()
        # Count blinks in the last 60 seconds
        recent_blinks = [t for t in self.blink_timestamps if current_time - t <= 60.0]
        
        return len(recent_blinks)
    
    def update_plot_data(self, ear_left, ear_right):
        """Update data for live plotting with 5-second intervals"""
        current_time = time.time()
        avg_ear = (ear_left + ear_right) / 2.0
        
        # Initialize plotting if not started
        if self.plot_start_time is None:
            self.plot_start_time = current_time
            self.current_interval_start = current_time
            
        # Add data to current interval
        self.current_interval_ears.append(avg_ear)
        
        # Check if we need to complete the current interval
        if current_time - self.current_interval_start >= self.plot_interval:
            self.complete_interval(current_time)
            
    def complete_interval(self, current_time):
        """Complete the current 5-second interval and add data point"""
        if not self.current_interval_ears:
            return
            
        # Calculate averages for this interval
        interval_ear_avg = sum(self.current_interval_ears) / len(self.current_interval_ears)
        
        # Calculate BPM for this interval (blinks in last 5 seconds * 12 to get per minute)
        interval_bpm = len(self.current_interval_blinks) * 12
        
        # Store the data point
        interval_center_time = self.current_interval_start + (self.plot_interval / 2)
        self.window_times.append(interval_center_time)
        self.window_ear.append(interval_ear_avg)
        self.window_bpm.append(interval_bpm)
        
        # Check if we have a full 5-minute window (60 points)
        if len(self.window_times) > 60:
            self.save_plot_window()
            # Remove oldest data point to maintain window size
            self.window_times.pop(0)
            self.window_ear.pop(0)
            self.window_bpm.pop(0)
            
        # Reset for next interval
        self.current_interval_start = current_time
        self.current_interval_ears = []
        self.current_interval_blinks = []
        
    def save_plot_window(self):
        """Save the current 5-minute window plot"""
        if not self.window_times:
            return
            
        try:
            # Create a temporary figure for saving
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Convert times to relative seconds from start
            start_time = self.plot_start_time
            relative_times = [(t - start_time) for t in self.window_times]
            
            # Plot BPM
            ax1.plot(relative_times, self.window_bpm, 'b-', linewidth=2, marker='o', markersize=3)
            ax1.set_title(f'Blinks Per Minute (BPM) - 5 Minute Window #{self.plot_save_counter + 1}')
            ax1.set_ylabel('BPM')
            ax1.set_ylim(0, max(30, max(self.window_bpm) * 1.1) if self.window_bpm else 30)
            ax1.grid(True, alpha=0.3)
            
            # Plot EAR
            ax2.plot(relative_times, self.window_ear, 'g-', linewidth=2, marker='o', markersize=3)
            ax2.axhline(y=self.EAR_THRESHOLD, color='r', linestyle='--', label='Threshold')
            ax2.set_title(f'Eye Aspect Ratio (EAR) - 5 Minute Window #{self.plot_save_counter + 1}')
            ax2.set_ylabel('EAR')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylim(0.1, 0.4)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"blink_plot_window_{self.plot_save_counter + 1}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.plot_save_counter += 1
            print(f"Saved plot window: {filename}")
            
        except Exception as e:
            print(f"Error saving plot window: {e}")
            
    def record_blink_for_interval(self):
        """Record a blink for the current interval"""
        self.current_interval_blinks.append(time.time())
    
    def update_plots(self):
        """Update the live plots (must be called from main thread)"""
        if not self.window_times:
            return
            
        try:
            # Check if the figure is still valid
            if not plt.fignum_exists(self.fig.number):
                print("Plot window was closed. Disabling live plotting.")
                return
                
            # Convert timestamps to relative seconds
            start_time = self.plot_start_time if self.plot_start_time else self.window_times[0]
            relative_times = [(t - start_time) for t in self.window_times]
            
            # Update BPM plot
            self.ax1.clear()
            self.ax1.plot(relative_times, self.window_bpm, 'b-', linewidth=2, marker='o', markersize=4)
            current_bpm = self.window_bpm[-1] if self.window_bpm else 0
            self.ax1.set_title(f'Blinks Per Minute (BPM) - Current: {current_bpm:.1f}')
            self.ax1.set_ylabel('BPM')
            self.ax1.set_ylim(0, max(30, max(self.window_bpm) * 1.1) if self.window_bpm else 30)
            self.ax1.grid(True, alpha=0.3)
            
            # Set x-axis to show 5-minute window
            if relative_times:
                self.ax1.set_xlim(max(0, relative_times[-1] - 300), relative_times[-1] + 10)
            
            # Update EAR plot
            self.ax2.clear()
            self.ax2.plot(relative_times, self.window_ear, 'g-', linewidth=2, marker='o', markersize=4)
            self.ax2.axhline(y=self.EAR_THRESHOLD, color='r', linestyle='--', label='Threshold')
            current_ear = self.window_ear[-1] if self.window_ear else 0
            self.ax2.set_title(f'Eye Aspect Ratio (EAR) - Current: {current_ear:.3f}')
            self.ax2.set_ylabel('EAR')
            self.ax2.set_xlabel('Time (seconds)')
            self.ax2.set_ylim(0.1, 0.4)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()
            
            # Set x-axis to show 5-minute window
            if relative_times:
                self.ax2.set_xlim(max(0, relative_times[-1] - 300), relative_times[-1] + 10)
            
            plt.tight_layout()
            
            # More robust drawing approach
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except:
                # Fallback if canvas operations fail
                plt.draw()
                
        except Exception as e:
            print(f"Error updating plots (non-fatal): {e}")
            # Don't crash the program, just skip this update
    
    def run(self):
        """Main detection loop"""
        print("Initializing camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not access camera. Please check camera connection.")
            return
        
        # Optimize camera settings for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Optimal balance
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Optimal balance
        cap.set(cv2.CAP_PROP_FPS, 30)            # Set target FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer to minimize latency
            
        print("Starting blink detection...")
        print("Advanced 5-second interval plotting system:")
        print("  - Data averaged over 5-second intervals")
        print("  - 5-minute scrolling window (60 data points)")
        print("  - Plots saved automatically when window scrolls")
        print("Controls:")
        print("  'q' to quit")
        print("  'r' to reset counter") 
        print("  'c' to calibrate EAR threshold")
        print("  '+' to increase threshold by 0.01")
        print("  '-' to decrease threshold by 0.01")
        print("  's' to toggle YOLO skip frames (5/10/15/20)")
        print("  'p' to toggle live plotting on/off")
        print("  'd' to toggle debug mode")
        
        # Performance monitoring
        frame_time_start = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Detect face using YOLO (optimized - skip frames)
            self.frame_count += 1
            
            # Only run YOLO every N frames for performance
            if self.frame_count % self.yolo_skip_frames == 0 or self.last_valid_face_coords is None:
                face_region, face_coords = self.detect_face_yolo(frame)
                if face_coords is not None:
                    self.last_valid_face_coords = face_coords
                    self.face_coords_valid_frames = 0
                else:
                    self.face_coords_valid_frames += 1
            else:
                # Use last known face coordinates (but invalidate after too long)
                self.face_coords_valid_frames += 1
                if self.last_valid_face_coords is not None and self.face_coords_valid_frames < 60:  # Max 2 seconds
                    x1, y1, x2, y2 = self.last_valid_face_coords
                    face_region = frame[y1:y2, x1:x2]
                    face_coords = self.last_valid_face_coords
                else:
                    face_region, face_coords = None, None
                    self.last_valid_face_coords = None
            
            if face_region is not None:
                # Process face with MediaPipe
                rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                
                # Get face region dimensions for MediaPipe
                face_h, face_w = face_region.shape[:2]
                
                # Process with MediaPipe (this fixes the NORM_RECT warning)
                results = self.face_mesh.process(rgb_face)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Calculate EAR for both eyes
                    ear_left = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES)
                    ear_right = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
                    avg_ear = (ear_left + ear_right) / 2.0
                    
                    # Store EAR history for smoothing
                    self.ear_history.append(avg_ear)
                    
                    # Minimal console output (only in debug mode)
                    if self.debug_mode and len(self.ear_history) % 150 == 0:  # Every 5 seconds
                        print(f"EAR: {avg_ear:.3f}, Threshold: {self.EAR_THRESHOLD}, FPS: {self.current_fps}")
                    
                    # Blink detection logic
                    if avg_ear < self.EAR_THRESHOLD:
                        self.eye_closed_counter += 1
                        if self.debug_mode and self.eye_closed_counter == 1:  # Only debug output
                            print(f"Eye closure detected: EAR={avg_ear:.3f}")
                    else:
                        # Eyes opened - check if we had a valid blink
                        if self.eye_closed_counter >= self.BLINK_CONSECUTIVE_FRAMES:
                            self.blink_counter += 1
                            self.blink_timestamps.append(time.time())
                            self.record_blink_for_interval()  # Record for interval plotting
                            if self.debug_mode:
                                print(f"Blink detected! Total: {self.blink_counter} (closed for {self.eye_closed_counter} frames)")
                        elif self.debug_mode and self.eye_closed_counter > 0:
                            print(f"Brief eye closure ignored (only {self.eye_closed_counter} frames)")
                        self.eye_closed_counter = 0
                    
                    # Update plot data
                    self.update_plot_data(ear_left, ear_right)
            
            # Create fixed-size display frame
            display_frame, face_display_coords = self.create_fixed_display_frame(face_region, frame)
            display_h, display_w = display_frame.shape[:2]
            
            # Draw eye landmarks on the display frame (only if debug mode is on)
            if (face_region is not None and 
                'results' in locals() and results.multi_face_landmarks and 
                face_display_coords is not None):
                
                landmarks = results.multi_face_landmarks[0].landmark
                start_x, start_y, face_w, face_h = face_display_coords
                
                for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
                    landmark = landmarks[idx]
                    x = int(landmark.x * face_w) + start_x
                    y = int(landmark.y * face_h) + start_y
                    cv2.circle(display_frame, (x, y), 2, (0, 255, 255), -1)
            
            # Update plots from main thread (reduce frequency for better performance)
            if self.frame_count % 20 == 0 and self.fig is not None:  # Every 20 frames
                self.update_plots()
            
            # Calculate and display FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Display information on frame
            current_bpm = self.calculate_bpm()
            current_ear = self.ear_history[-1] if self.ear_history else 0.0
            
            # Create expanded frame with info section below (using fixed dimensions)
            info_height = 140  # Fixed height for info section
            expanded_frame = np.zeros((display_h + info_height, display_w, 3), dtype=np.uint8)
            expanded_frame[:display_h, :display_w] = display_frame
            
            # Create dark info section
            info_section = expanded_frame[display_h:display_h+info_height, :display_w]
            info_section.fill(30)  # Dark gray background
            
            # Add a separator line
            cv2.line(expanded_frame, (0, display_h), (display_w, display_h), (100, 100, 100), 2)
            
            # Info text with better formatting (adjusted for smaller display)
            # Calculate interval progress
            interval_progress = ""
            if self.current_interval_start is not None:
                elapsed = time.time() - self.current_interval_start
                remaining = max(0, self.plot_interval - elapsed)
                interval_progress = f"{remaining:.1f}s"
            
            info_data = [
                ("FPS", f"{self.current_fps}"),
                ("Blinks", f"{self.blink_counter}"),
                ("BPM", f"{current_bpm:.1f}"),
                ("EAR", f"{current_ear:.3f}"),
                ("Thresh", f"{self.EAR_THRESHOLD:.3f}"),  # Shortened label
                ("Skip", f"{self.yolo_skip_frames}"),     # Shortened label
                ("Debug", "ON" if self.debug_mode else "OFF"),
                ("Plot", "ON" if self.fig is not None else "OFF"),
                ("Interval", interval_progress),
                ("Windows", f"{self.plot_save_counter}")
            ]
            
            # Display info in two columns (adjusted for smaller display)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.4  # Smaller font for fixed display
            thickness = 1
            
            for i, (label, value) in enumerate(info_data):
                col = i % 2  # 0 or 1 for two columns
                row = i // 2  # Row number
                
                x_pos = 10 + col * (display_w // 2)  # Adjusted spacing
                y_pos = display_h + 20 + row * 25    # Adjusted spacing
                
                # Label in lighter color
                cv2.putText(expanded_frame, f"{label}:", (x_pos, y_pos), 
                          font, font_scale, (200, 200, 200), thickness)
                
                # Value in white, offset to the right
                cv2.putText(expanded_frame, value, (x_pos + 60, y_pos), 
                          font, font_scale, (255, 255, 255), thickness)
            
            # Display the expanded frame
            cv2.imshow('Blink Detector', expanded_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.blink_counter = 0
                self.blink_timestamps.clear()
                print("Blink counter reset!")
            elif key == ord('c'):
                # Calibrate threshold based on current EAR
                if self.ear_history:
                    avg_recent_ear = sum(list(self.ear_history)[-30:]) / min(30, len(self.ear_history))
                    self.EAR_THRESHOLD = avg_recent_ear * 0.8  # Set to 80% of current EAR
                    print(f"Threshold calibrated to: {self.EAR_THRESHOLD:.3f}")
            elif key == ord('+') or key == ord('='):
                self.EAR_THRESHOLD += 0.01
                print(f"Threshold increased to: {self.EAR_THRESHOLD:.3f}")
            elif key == ord('-') or key == ord('_'):
                self.EAR_THRESHOLD = max(0.1, self.EAR_THRESHOLD - 0.01)
                print(f"Threshold decreased to: {self.EAR_THRESHOLD:.3f}")
            elif key == ord('s'):
                # Toggle YOLO skip frames for performance tuning
                if self.yolo_skip_frames == 5:
                    self.yolo_skip_frames = 10
                elif self.yolo_skip_frames == 10:
                    self.yolo_skip_frames = 15
                elif self.yolo_skip_frames == 15:
                    self.yolo_skip_frames = 20
                else:
                    self.yolo_skip_frames = 5
                print(f"YOLO skip frames set to: {self.yolo_skip_frames}")
            elif key == ord('p'):
                # Toggle plotting on/off
                if self.fig is not None:
                    plt.close(self.fig)
                    self.fig = None
                    self.ax1 = None
                    self.ax2 = None
                    print("Live plotting disabled")
                else:
                    print("Reinitializing live plotting...")
                    self.setup_live_plot()
                    if self.fig is not None:
                        print("Live plotting enabled")
                    else:
                        print("Failed to reinitialize plotting")
            elif key == ord('d'):
                # Toggle debug mode
                self.debug_mode = not self.debug_mode
                print(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Safely close matplotlib
        try:
            if self.fig is not None:
                plt.close(self.fig)
            plt.close('all')
        except:
            pass  # Ignore cleanup errors
            
        print("Blink detector stopped.")

def main():
    """Main function to run the blink detector"""
    try:
        detector = BlinkDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\nStopping blink detector...")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a camera connected and the required packages installed.")

if __name__ == "__main__":
    main()