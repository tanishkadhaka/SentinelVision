# SentinelVision — Multi-Zone Vision-Based Intrusion Detection

**SentinelVision** is a computer vision based surveillance system that detects and tracks people and vehicles entering or leaving user defined polygon zones in video.

The system combines **YOLOv8 object detection, SORT multi-object tracking, OpenCV polygon analysis, and event logging** to convert raw video into meaningful zone-based events.

An annotated output video is generated showing detected objects, tracking IDs, surveillance zones, and detected entry/exit events.

---

## Key Features

* YOLOv8 based object detection
* Multi-object tracking using SORT
* Persistent tracking IDs across video frames
* User-defined polygon surveillance zones
* Automatic entry and exit detection
* Person and vehicle detection
* OpenCV based point-in-polygon testing
* Annotated output video generation
* Timestamped event logging
* CSV based event history
* Debug mode for detection and tracking analysis
* Headless video processing using `--no-display`
* Configurable detection and tracking parameters

---

## System Architecture

Each video frame passes through the following computer vision pipeline:

```text
Input Video
     │
     ▼
YOLOv8 Object Detection
     │
     ▼
Detection Filtering
     │
     ▼
SORT Multi-Object Tracking
     │
     ▼
Stable Object IDs
     │
     ▼
Polygon Zone Analysis
     │
     ▼
Entry / Exit Detection
     │
     ├──────────────► Event Logger
     │                     │
     │                     ▼
     │                 events.csv
     │
     ▼
Annotated Output Video
```

---

## How It Works

### 1. Object Detection

Each frame is processed using **YOLOv8** to detect relevant objects.

For every detection, the system obtains:

* Bounding box coordinates
* Object class
* Detection confidence

The detector can be configured to focus on relevant person and vehicle classes.

---

### 2. Multi-Object Tracking

The detections are passed to a **SORT (Simple Online and Realtime Tracking)** tracker.

SORT combines:

* Kalman filtering
* Intersection over Union (IoU)
* Hungarian algorithm based matching

This allows detected objects to maintain a stable tracking ID across consecutive frames.

Example:

```text
Person  → ID 7
Vehicle → ID 12
Person  → ID 15
```

The tracking ID allows the system to determine whether the same object has moved into or out of a surveillance zone.

---

### 3. User-Defined Surveillance Zones

Users can create custom polygon zones directly on the video.

For example:

```text
┌─────────────────────────────────────────────┐
│                                             │
│        ┌─────────────────────────┐          │
│        │                         │          │
│        │     RESTRICTED ZONE     │          │
│        │                         │          │
│        └─────────────────────────┘          │
│                                             │
│                    ┌──────────────┐         │
│                    │ VEHICLE ZONE │         │
│                    └──────────────┘         │
│                                             │
└─────────────────────────────────────────────┘
```

Multiple zones can be created within the same video.

---

### 4. Polygon Zone Analysis

For every tracked object, the center point of its bounding box is calculated.

OpenCV's `pointPolygonTest` is then used to determine whether the point lies inside each user-defined polygon.

The system maintains the object's previous zone membership to identify transitions.

```text
Outside → Inside
        ↓
      ENTER

Inside → Outside
        ↓
       EXIT
```

This allows the system to detect actual zone crossings rather than repeatedly reporting an object that remains inside a zone.

---

### 5. Event Logging

When an entry or exit event occurs, SentinelVision records:

```text
timestamp
object_id
zone
event
```

Events are:

* Printed to the console
* Displayed on the processed video
* Saved to `logs/events.csv`

Example:

```text
[14:32:18] Object 12 ENTERED Restricted Area
[14:33:02] Object 12 EXITED Restricted Area
```

---

# Output Video

SentinelVision generates an annotated output video containing:

* Detected objects
* Bounding boxes
* Tracking IDs
* User-defined zones
* Zone status
* Entry/exit events
* Detection information

The output video makes it possible to visually inspect the detection and tracking pipeline and verify the generated events.

---

# Project Structure

```text
SentinelVision/
│
├── main.py
├── detector.py
├── sort_tracker.py
├── zone_logic.py
├── zone_drawer.py
├── event_logger.py
├── config.py
├── requirements.txt
├── zones.json
│
├── input/
│   └── input_video.mp4
│
├── output/
│   └── output_video.mp4
│
├── logs/
│   └── events.csv
│
└── README.md
```

### File Responsibilities

| File              | Purpose                                          |
| ----------------- | ------------------------------------------------ |
| `main.py`         | Main video processing pipeline                   |
| `detector.py`     | YOLOv8 detection wrapper                         |
| `sort_tracker.py` | SORT multi-object tracker                        |
| `zone_logic.py`   | Polygon zones and entry/exit logic               |
| `zone_drawer.py`  | Interactive zone creation                        |
| `event_logger.py` | Event logging and video event overlay            |
| `config.py`       | Detection, tracking and visualization parameters |
| `zones.json`      | Saved user-defined surveillance zones            |
| `logs/events.csv` | Generated zone events                            |

---

# Tech Stack

### Computer Vision

* Python
* YOLOv8
* OpenCV
* SORT
* Kalman Filter
* Hungarian Algorithm
* IoU based object association
* Polygon geometry

### Development

* Git
* GitHub
* Python virtual environments
* CSV based logging

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SentinelVision.git
cd SentinelVision
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The first run of the application downloads the required YOLOv8 model weights through Ultralytics.

---

# Usage

## 1. Create Surveillance Zones

Run:

```bash
python zone_drawer.py --video path\to\video.mp4 --zones zones.json
```

### Controls

| Input       | Action            |
| ----------- | ----------------- |
| Left Click  | Add polygon point |
| Right Click | Undo last point   |
| Enter / `c` | Close polygon     |
| `s`         | Save zones        |
| `r`         | Clear zones       |
| `q` / Esc   | Quit              |

Multiple surveillance zones can be created before saving them to `zones.json`.

---

## 2. Run Detection and Tracking

Run:

```bash
python main.py --video path\to\video.mp4 --zones zones.json
```

The system will:

1. Read the input video
2. Detect objects using YOLOv8
3. Track detected objects using SORT
4. Calculate object positions
5. Test positions against polygon zones
6. Detect entry and exit events
7. Generate an annotated output video
8. Record events in `logs/events.csv`

---

## 3. Headless Processing

For processing without displaying the video window:

```bash
python main.py --video path\to\video.mp4 --zones zones.json --no-display
```

This is useful for batch processing or environments without a graphical display.

---

## 4. Debug Mode

Run:

```bash
python main.py --video path\to\video.mp4 --zones zones.json --debug
```

Debug mode provides additional information about:

* Raw YOLO detections
* Detection confidence
* Object classes
* SORT configuration
* New tracks
* Lost tracks
* Zone membership
* Per-frame tracking observations

The information is saved to:

```text
logs/debug.csv
```

This can be used to investigate unexpected tracking IDs, missed detections, and unexpected zone events.

---

# Event Output

Generated events follow the structure:

```csv
timestamp,object_id,zone,event
14:32:18,12,Restricted_Area,ENTER
14:33:02,12,Restricted_Area,EXIT
14:35:41,18,Vehicle_Zone,ENTER
```

This provides a structured record that can later be used for analytics or integration with other monitoring systems.

---

# Computer Vision Concepts Demonstrated

### Object Detection

YOLOv8 is used to locate and classify objects within video frames.

### Multi-Object Tracking

SORT associates detections across consecutive frames and assigns tracking IDs.

### Kalman Filtering

The tracker uses Kalman filtering to estimate object motion and maintain tracks through short detection gaps.

### IoU Matching

Intersection over Union is used to measure overlap between detections and predicted tracks.

### Hungarian Algorithm

Hungarian matching is used to associate detections with existing tracks.

### Polygon Geometry

OpenCV polygon testing determines whether an object's position falls within a user-defined surveillance region.

### Event-Based Video Analytics

The project goes beyond simply detecting objects by converting visual observations into meaningful events:

```text
Object detected
      ↓
Object tracked
      ↓
Object enters zone
      ↓
ENTER event generated
      ↓
Event logged
```

---

# Real-World Considerations

Video analytics systems operating in real environments can encounter:

* Partial occlusion
* Object overlap
* Motion
* Changing lighting
* Motion blur
* Detection confidence variations
* Small or distant objects
* Objects near zone boundaries
* Temporary missed detections

SentinelVision includes configurable detection/tracking parameters and debug logging to make the pipeline easier to inspect and tune under different video conditions.

---

# Limitations

## Long Occlusions

If an object is not detected for longer than the configured SORT `max_age`, its track can be removed.

When the object reappears, SORT may assign a new tracking ID.

Standard SORT does not use appearance-based re-identification.

A future version could use Deep SORT or another appearance-aware tracker to improve identity persistence after long occlusions.

## Zone Boundary Flicker

Objects positioned directly on a zone boundary can move between inside and outside states because of small changes in their detected bounding box.

A future improvement would be to introduce a hysteresis margin around zone boundaries.

## Detection Quality

Detection performance can vary depending on:

* Video resolution
* Lighting
* Camera angle
* Object size
* Occlusion
* Motion blur
* Detection confidence

---

# Future Improvements

Potential extensions include:

* Deep SORT based appearance tracking
* Re-identification after long occlusions
* Zone-specific alert levels
* REST API based alerts
* Real-time monitoring dashboard
* Automatic camera stream support
* Detection speed and accuracy benchmarking
* Confidence threshold optimization
* Multi-camera tracking
* More advanced video event classification

---

# Applications

The pipeline can be adapted for:

* Restricted-area monitoring
* Perimeter monitoring
* Industrial safety
* Parking and vehicle zones
* Warehouse monitoring
* Campus surveillance
* Access monitoring
* Video-based security analytics

---

# Project Objective

The primary objective of SentinelVision is to build a practical computer vision pipeline that converts video into structured, actionable events.

The project combines:

```text
Object Detection
       +
Object Tracking
       +
Spatial Reasoning
       +
Event Detection
       +
Video Analytics
       +
Structured Logging
```

This demonstrates the implementation of an end-to-end computer vision workflow rather than using object detection as an isolated component.

---

# Author

**Tanishka Dhaka**

B.Tech — Artificial Intelligence & Data Science

Computer Vision | Machine Learning | Data Science

---

# License

This project is intended for educational and research purposes.
