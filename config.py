"""Central configuration for ZoneGuard."""

# Files
DEFAULT_ZONES_FILE = "zones.json"
DEFAULT_LOG_CSV = "logs/events.csv"

# Detection
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45

# COCO class ids we care about: people + vehicles
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Tracking (SORT)
SORT_MAX_AGE = 15       # frames to keep a track alive without a matching detection
SORT_MIN_HITS = 3       # consecutive matches before a track is considered confirmed
SORT_IOU_THRESHOLD = 0.3

# Display
ZONE_COLOR = (0, 255, 255)       # yellow outline for zones
ZONE_FILL_ALPHA = 0.15
BOX_COLOR = (0, 200, 0)          # green detection/track boxes
ALERT_COLOR = (0, 0, 255)        # red highlight for objects currently inside a zone
EVENT_OVERLAY_MAX_LINES = 6
