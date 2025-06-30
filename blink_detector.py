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
            min_detection_confidence=0.7,  # Higher confidence = fewer false positives
            min_tracking_confidence=0.5
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
        self.yolo_skip_frames = 5  # Only run YOLO every 5 frames
        self.frame_count = 0
        self.last_valid_face_coords = None
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Live plotting setup
        self.setup_live_plot()
        
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
            
            # Data for plotting
            self.plot_times = deque(maxlen=300)
            self.plot_bpm = deque(maxlen=300)
            self.plot_ear = deque(maxlen=300)
            
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
            
            # Debug: Print calculations occasionally
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 0
                
            if self.debug_counter % 120 == 0:  # Every 120 frames (4 seconds) - reduced frequency
                print(f"DEBUG EAR Calculation:")
                print(f"  Eye points shape: {eye_points.shape}")
                print(f"  Vertical_1: {vertical_1:.4f}")
                print(f"  Vertical_2: {vertical_2:.4f}")
                print(f"  Horizontal: {horizontal:.4f}")
                print(f"  EAR: {ear:.4f}")
                print(f"  Eye points: {eye_points}")
            
            return ear
                
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.0
    
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
        """Update data for live plotting"""
        current_time = time.time()
        avg_ear = (ear_left + ear_right) / 2.0
        current_bpm = self.calculate_bpm()
        
        self.plot_times.append(current_time)
        self.plot_ear.append(avg_ear)
        self.plot_bpm.append(current_bpm)
    
    def update_plots(self):
        """Update the live plots (must be called from main thread)"""
        if len(self.plot_times) < 2:
            return
            
        try:
            # Check if the figure is still valid
            if not plt.fignum_exists(self.fig.number):
                print("Plot window was closed. Disabling live plotting.")
                return
                
            # Convert timestamps to relative seconds
            start_time = self.plot_times[0]
            relative_times = [(t - start_time) for t in self.plot_times]
            
            # Update BPM plot
            self.ax1.clear()
            self.ax1.plot(relative_times, list(self.plot_bpm), 'b-', linewidth=2)
            self.ax1.set_title(f'Blinks Per Minute (BPM) - Current: {self.plot_bpm[-1]:.1f}')
            self.ax1.set_ylabel('BPM')
            self.ax1.set_ylim(0, 30)
            self.ax1.grid(True, alpha=0.3)
            
            # Update EAR plot
            self.ax2.clear()
            self.ax2.plot(relative_times, list(self.plot_ear), 'g-', linewidth=2)
            self.ax2.axhline(y=self.EAR_THRESHOLD, color='r', linestyle='--', label='Threshold')
            self.ax2.set_title(f'Eye Aspect Ratio (EAR) - Current: {self.plot_ear[-1]:.3f}')
            self.ax2.set_ylabel('EAR')
            self.ax2.set_xlabel('Time (seconds)')
            self.ax2.set_ylim(0.1, 0.4)
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()
            
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduce from default (usually 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduce from default (usually 720)
        cap.set(cv2.CAP_PROP_FPS, 30)            # Set target FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffer to minimize latency
            
        print("Starting blink detection...")
        print("Controls:")
        print("  'q' to quit")
        print("  'r' to reset counter") 
        print("  'c' to calibrate EAR threshold")
        print("  '+' to increase threshold by 0.01")
        print("  '-' to decrease threshold by 0.01")
        print("  's' to toggle YOLO skip frames (5/10/15)")
        print("  'p' to toggle live plotting on/off")
        
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
            else:
                # Use last known face coordinates
                if self.last_valid_face_coords is not None:
                    x1, y1, x2, y2 = self.last_valid_face_coords
                    face_region = frame[y1:y2, x1:x2]
                    face_coords = self.last_valid_face_coords
                else:
                    face_region, face_coords = None, None
            
            if face_region is not None:
                # Process face with MediaPipe
                rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_face)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # Calculate EAR for both eyes
                    ear_left = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES)
                    ear_right = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
                    avg_ear = (ear_left + ear_right) / 2.0
                    
                    # Store EAR history for smoothing
                    self.ear_history.append(avg_ear)
                    
                    # Debug: Print EAR values occasionally (reduced frequency)
                    if len(self.ear_history) % 90 == 0:  # Every 90 frames (about 3 seconds)
                        print(f"Current EAR: {avg_ear:.3f}, Threshold: {self.EAR_THRESHOLD}, FPS: {self.current_fps}")
                    
                    # Blink detection logic
                    if avg_ear < self.EAR_THRESHOLD:
                        self.eye_closed_counter += 1
                        if self.eye_closed_counter == 1:  # First frame of potential blink
                            print(f"Eye closure detected: EAR={avg_ear:.3f}")
                    else:
                        # Eyes opened - check if we had a valid blink
                        if self.eye_closed_counter >= self.BLINK_CONSECUTIVE_FRAMES:
                            self.blink_counter += 1
                            self.blink_timestamps.append(time.time())
                            print(f"Blink detected! Total: {self.blink_counter} (was closed for {self.eye_closed_counter} frames)")
                        elif self.eye_closed_counter > 0:
                            print(f"Brief eye closure ignored (only {self.eye_closed_counter} frames)")
                        self.eye_closed_counter = 0
                    
                    # Update plot data
                    self.update_plot_data(ear_left, ear_right)
                    
                    # Draw face rectangle
                    x1, y1, x2, y2 = face_coords
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw eye landmarks on the main frame
                    for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
                        landmark = landmarks[idx]
                        x = int(landmark.x * (x2 - x1)) + x1
                        y = int(landmark.y * (y2 - y1)) + y1
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # Update plots from main thread (every 15 frames to reduce lag)
                if len(self.plot_times) % 15 == 0 and self.fig is not None:  # Check if plotting is available
                    self.update_plots()
                    
            else:
                # No face detected
                if self.face_detection_failures > 30:  # After 1 second at 30fps
                    cv2.putText(frame, "No face detected", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate and display FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Display information on frame
            current_bpm = self.calculate_bpm()
            current_ear = self.ear_history[-1] if self.ear_history else 0.0
            
            info_text = [
                f"FPS: {self.current_fps}",
                f"Blinks: {self.blink_counter}",
                f"BPM: {current_bpm:.1f}",
                f"EAR: {current_ear:.3f}",
                f"Threshold: {self.EAR_THRESHOLD}"
            ]
            
            for i, text in enumerate(info_text):
                y_pos = 30 + i * 30
                cv2.putText(frame, text, (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Blink Detector', frame)
            
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