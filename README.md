# Real-Time-Detection-of-Object-Missing-and-New-Object-Placement-in-Video

# 🚀 Overview
# Welcome to the Universal Object Tracker!
This project is a real-time video analytics pipeline that can:  
🔍 *Detect when an object goes missing from the scene*  
🆕 *Detect when a new object enters the scene*

Built for the *Samajh.ai ML Engineer Intern* evaluation, it uses:  
*YOLOv8* (for detection)  
*DeepSORT* (for tracking)  
All packaged neatly in *Python + Docker*!  

# ✨ Features
* *Real-Time Object Detection* with YOLOv8 (large model for higher accuracy)  
* *Multi-Object Tracking* using DeepSORT (MobileNet embedder)  
* *Event Detection:*
  * Flags missing objects  
  * Highlights new objects  
* *Visual Feedback:*
  * Color-coded bounding boxes  
  * Blinking red cross (X) for missing objects  
  * Real-time object counts per class  
* *Performance Monitoring:*
  * FPS tracking  
  * Processing time tracking  
* *Dockerized Setup* for smooth and reproducible environments  

# 📸 Sample
<img width="1280" alt="image" src="https://github.com/user-attachments/assets/6730e42e-93a6-4fea-b1d6-8e46996b5ec3" />

# 💻 Hardware Used
*Device:* MacBook Pro 13-inch (2022)  
*Processor:* Apple M2 chip (8-core CPU, 10-core GPU)  
*RAM:* 8 GB  
*OS:* macOS Sonoma  

# 🏃‍♂️ Quick Start
**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/universal-object-tracker.git
cd universal-object-tracker
```
**2. Install Requirements**
```bash
pip install -r requirements.txt
```
**3. Run the Tracker (Directly)**
```bash
python detect_missing_new_objects.py
```
**4. Or Run with Docker**
```bash
docker build -t universal-object-tracker .
docker run -it -v $(pwd)/sample_video2.move:/app/sample_video2.move -v $(pwd)/outputs:/app/outputs universal-object-tracker
```

# 📝 Usage
- Place your input video (e.g., sample_video2.move) in the project folder.
- The script automatically saves:
 - The processed output video → outputs/output.mp4
 - Sample output frames (optional if you want to add)

# ⚡ Performance
- FPS Achieved: ~1.54 FPS (YOLOv8 large, MacBook Pro M2)
- For faster results, try using a smaller YOLOv8 model (e.g., YOLOv8n) or run on a dedicated GPU.

# 🧠 How It Works
- **Detection:** YOLOv8 detects objects in each frame.
- **Tracking:** DeepSORT assigns consistent IDs across frames.
- **Event Logic:**
 - If an object disappears → flagged as missing
 - If a new object appears → flagged as new
- **Visualization:**
 - Bounding boxes
 - Object labels
 - Real-time counters drawn over video frames
