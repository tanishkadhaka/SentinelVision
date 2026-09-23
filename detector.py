"""YOLOv8 wrapper: runs detection on a frame and returns filtered boxes."""
import numpy as np
from ultralytics import YOLO

import config


def suppress_contained_boxes(boxes, classes, containment_thresh=0.8):
    """
    Standard IoU-based NMS misses a same-class detection that is mostly
    *contained inside* a much larger one (e.g. a hand/limb sub-region of a
    person separately flagged as its own low-confidence "person" box) --
    IoU stays low under a large size mismatch even at ~100% containment.
    Drop the smaller box when (intersection / smaller_box_area) exceeds
    containment_thresh, keeping the larger/higher-confidence one.
    """
    n = len(boxes)
    if n <= 1:
        return boxes, classes

    order = sorted(range(n), key=lambda i: boxes[i][4], reverse=True)
    suppressed = set()
    keep = []

    for idx_a in order:
        if idx_a in suppressed:
            continue
        keep.append(idx_a)
        ax1, ay1, ax2, ay2 = boxes[idx_a][:4]
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)

        for idx_b in order:
            if idx_b == idx_a or idx_b in suppressed or classes[idx_b] != classes[idx_a]:
                continue
            bx1, by1, bx2, by2 = boxes[idx_b][:4]
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            if area_b == 0 or area_b >= area_a:
                continue

            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

            if inter / area_b >= containment_thresh:
                suppressed.add(idx_b)

    keep = sorted(keep)
    return boxes[keep], [classes[i] for i in keep]


class Detector:
    def __init__(self, weights=config.MODEL_WEIGHTS,
                 conf=config.CONF_THRESHOLD,
                 iou=config.IOU_THRESHOLD,
                 target_classes=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.target_classes = target_classes or config.TARGET_CLASSES

    def detect(self, frame):
        """
        Returns np.ndarray of shape (N, 5): [x1, y1, x2, y2, conf]
        for detections whose class is in target_classes.
        Class labels for each row are dropped here since the tracker only
        needs boxes; call detect_with_classes() if you need class ids too.
        """
        boxes, _ = self.detect_with_classes(frame)
        return boxes

    def detect_with_classes(self, frame):
        results = self.model.predict(
            frame, conf=self.conf, iou=self.iou, verbose=False
        )[0]

        boxes_out = []
        classes_out = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            boxes_out.append([x1, y1, x2, y2, conf])
            classes_out.append(cls_id)

        boxes_out = np.array(boxes_out) if boxes_out else np.empty((0, 5))
        boxes_out, classes_out = suppress_contained_boxes(boxes_out, classes_out)
        return boxes_out, classes_out
