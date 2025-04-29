import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class UniversalObjectTracker:
    def __init__(self, confidence_threshold=0.4, tracking_classes=None, output_path='output.mp4', fps=30):
        """
        Initialize the universal object tracker.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detection (0-1)
            tracking_classes (list): List of class names to track (None for all classes)
            output_path (str): Path to save output video
            fps (int): Frames per second for output video
        """
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8l.pt')  # Using large model for better accuracy
        self.model.conf = confidence_threshold  # Set model confidence
        self.model.iou = 0.5  # Set IOU threshold higher for better filtering
        self.model.agnostic_nms = False
        self.model.max_det = 300  # Allow more detections
        
        self.conf_thresh = confidence_threshold
        self.tracking_classes = tracking_classes
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_iou_distance=0.7,
            max_age=30,
            n_init=3,
            nn_budget=100,
            embedder="mobilenet",
            half=True,
            bgr=True
        )
        
        # Tracking data
        self.tracked_objects = {}
        self.missing_objects = {}
        self.new_objects = set()
        self.class_counts = defaultdict(int)
        self.total_count = 0
        
        # Configuration
        self.missing_threshold = 10
        self.new_threshold = 5
        self.stability_threshold = 10
        self.max_position_variation = 20
        
        # Visualization settings
        self.missing_dot_duration = 60
        self.blink_interval = 10
        self.colors = self._generate_colors()

        # Video writer setup
        self.output_path = output_path
        self.writer = None
        self.fps = fps

    def _generate_colors(self):
        return {
            'person': (0, 255, 0),
            'car': (255, 0, 0),
            'truck': (0, 0, 255),
            'bus': (255, 255, 0),
            'motorcycle': (0, 255, 255),
            'bicycle': (255, 0, 255),
            'default': (200, 200, 200)
        }

    def _get_class_color(self, class_name):
        return self.colors.get(class_name.lower(), self.colors['default'])

    def process_frame(self, frame):
        # Initialize writer
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        
        results = self.model(frame, verbose=False)[0]
        class_names = results.names
        
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            class_id = int(cls)
            class_name = class_names[class_id]
            if self.tracking_classes and class_name.lower() not in [c.lower() for c in self.tracking_classes]:
                continue
            if float(conf) >= self.conf_thresh:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections.append(([float(x1), float(y1), float(w), float(h)], float(conf), class_name))
        
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        self.class_counts = defaultdict(int)
        self.total_count = 0
        current_objects = {}
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            class_name = track.det_class if hasattr(track, 'det_class') else 'object'
            
            current_objects[track_id] = {
                'bbox': (x1, y1, x2, y2),
                'class': class_name,
                'center': center
            }
            
            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = {
                    'history': [],
                    'class': class_name
                }
            
            self.tracked_objects[track_id]['history'].append(center)
            if len(self.tracked_objects[track_id]['history']) > self.stability_threshold:
                self.tracked_objects[track_id]['history'].pop(0)
            
            self.class_counts[class_name] += 1
            self.total_count += 1
            
            color = self._get_class_color(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{class_name} {track_id}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        self._update_missing_objects(current_objects)
        self._update_new_objects(current_objects)
        self.tracked_objects = {k: v for k, v in self.tracked_objects.items() if k in current_objects}
        
        frame = self._visualize_missing_objects(frame)
        self._display_counts(frame)

        # Save frame
        self.writer.write(frame)
        
        return frame

    def _update_missing_objects(self, current_objects):
        missing_ids = set(self.tracked_objects.keys()) - set(current_objects.keys())
        print(f"Missing IDs: {missing_ids}") 
        for track_id in missing_ids:
            if track_id not in self.missing_objects:
                last_data = self.tracked_objects[track_id]
                self.missing_objects[track_id] = {
                    'last_seen': last_data['history'][-1] if last_data['history'] else (0, 0),
                    'class': last_data['class'],
                    'frames_missing': 1,
                    'frames_displayed': 0
                }
            else:
                self.missing_objects[track_id]['frames_missing'] += 1
        
        reappeared_ids = set(self.missing_objects.keys()) & set(current_objects.keys())
        for track_id in reappeared_ids:
            current_pos = current_objects[track_id]['center']
            last_missing_pos = self.missing_objects[track_id]['last_seen']
            
            distance = np.linalg.norm(np.array(current_pos) - np.array(last_missing_pos))
            
            if distance < self.max_position_variation:
                del self.missing_objects[track_id]
            else:
                self.new_objects.add(track_id)

    def _update_new_objects(self, current_objects):
        new_candidates = set(current_objects.keys()) - set(self.tracked_objects.keys())
        
        for track_id in new_candidates:
            if track_id not in self.new_objects:
                is_returning = False
                for missing_id, data in self.missing_objects.items():
                    last_pos = data['last_seen']
                    current_pos = current_objects[track_id]['center']
                    distance = np.linalg.norm(np.array(current_pos) - np.array(last_pos))
                    if distance < self.max_position_variation:
                        is_returning = True
                        if missing_id in self.missing_objects:
                            del self.missing_objects[missing_id]
                        break
                
                if not is_returning:
                    self.new_objects.add(track_id)

    def _visualize_missing_objects(self, frame):
        to_delete = []
        
        for track_id, data in self.missing_objects.items():
            if data['frames_missing'] >= self.missing_threshold:
                print(f"Visualizing missing {track_id}") 
                center = data['last_seen']
                frames_displayed = data['frames_displayed']
                class_name = data['class']
                
                if (frames_displayed // self.blink_interval) % 2 == 0:
                    color = (0, 0, 255)  # Red
                    # Draw red cross
                    cross_size = 20
                    cv2.line(frame, (center[0] - cross_size, center[1] - cross_size),
                                    (center[0] + cross_size, center[1] + cross_size), color, 3)
                    cv2.line(frame, (center[0] - cross_size, center[1] + cross_size),
                                    (center[0] + cross_size, center[1] - cross_size), color, 3)
                    cv2.putText(frame, f'Missing {class_name} {track_id}', 
                               (center[0]-40, center[1]-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                data['frames_displayed'] += 1
                
                if data['frames_displayed'] > self.missing_dot_duration:
                    to_delete.append(track_id)
        
        for track_id in to_delete:
            if track_id in self.missing_objects:
                del self.missing_objects[track_id]
        
        return frame

    def _display_counts(self, frame):
        cv2.putText(frame, f'Total Objects: {self.total_count}', 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset = 60
        for class_name, count in self.class_counts.items():
            color = self._get_class_color(class_name)
            cv2.putText(frame, f'{class_name}: {count}', (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30

    def release(self):
        """Release resources properly."""
        if self.writer:
            self.writer.release()




class PerformanceMonitor:
    """Performance monitoring utility."""
    def __init__(self):
        self.start_time = time.time()
        self.total_frames = 0
        self.total_process_time = 0

    def start_frame(self):
        self.frame_start_time = time.time()

    def end_frame(self):
        self.total_frames += 1
        frame_process_time = time.time() - self.frame_start_time
        self.total_process_time += frame_process_time

    def summarize(self):
        total_time_elapsed = time.time() - self.start_time
        if self.total_frames > 0:
            average_frame_time = self.total_process_time / self.total_frames
            fps = self.total_frames / total_time_elapsed
            print("\n=== Performance Summary ===")
            print(f"Total Time Elapsed: {total_time_elapsed:.2f} seconds")
            print(f"Total Frames Processed: {self.total_frames}")
            print(f"Average Frame Time: {average_frame_time*1000:.2f} ms")
            print(f"FPS: {fps:.2f}")
        else:
            print("No frames processed.")

def main():
    # Initialize tracker and performance monitor
    tracker = UniversalObjectTracker(confidence_threshold=0.5,output_path='my_output_video2.mp4')
    performance_monitor = PerformanceMonitor()
    
    # Video Input
    video_path = 'sample_video2.MOV'  # Replace with your video path
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
    
    print("Starting video processing...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        performance_monitor.start_frame()
        
        processed_frame = tracker.process_frame(frame)
        
        performance_monitor.end_frame()
        
        #cv2.imshow('Universal Object Tracker', processed_frame)
        out.write(processed_frame)
        cv2.imwrite('sample_output_frame.jpg', processed_frame)


        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    
    # Cleanup
    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    performance_monitor.summarize()
    print("Processing complete. Output saved to 'output.mp4'")

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # To prevent library errors
    main()
