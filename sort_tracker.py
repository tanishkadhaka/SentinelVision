"""
Minimal SORT (Simple Online and Realtime Tracking) implementation.

Each track is a constant-velocity Kalman filter over [cx, cy, scale, aspect_ratio].
Detections are associated to existing tracks frame-to-frame by IoU using the
Hungarian algorithm (scipy.optimize.linear_sum_assignment).

This is a from-scratch re-implementation of the well-known SORT algorithm
(Bewley et al., 2016) -- not a copy of any specific codebase.
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx,cy,s,r] (s=area, r=aspect ratio)."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h != 0 else 0.0
    return np.array([cx, cy, s, r]).reshape((4, 1))


def z_to_bbox(z):
    """[cx,cy,s,r] -> [x1,y1,x2,y2]."""
    s, r = max(z[2], 0.0), max(z[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w if w != 0 else 0.0
    x1 = z[0] - w / 2.0
    y1 = z[1] - h / 2.0
    x2 = z[0] + w / 2.0
    y2 = z[1] + h / 2.0
    return np.array([x1, y1, x2, y2])


def iou_batch(boxes_a, boxes_b):
    """Vectorized IoU between two arrays of [x1,y1,x2,y2] boxes."""
    boxes_a = np.expand_dims(boxes_a, 1)
    boxes_b = np.expand_dims(boxes_b, 0)

    xx1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    yy1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    xx2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    yy2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])
    union = area_a + area_b - intersection

    return np.where(union > 0, intersection / union, 0.0)


class KalmanBoxTracker:
    """A single tracked object with a constant-velocity Kalman filter."""

    _next_id = 1

    def __init__(self, bbox):
        # State: [cx, cy, s, r, vx, vy, vs] -- r assumed constant (no vr)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        for i in range(3):
            self.kf.F[i, i + 4] = 1.0  # constant velocity model
        self.kf.H = np.zeros((4, 7))
        for i in range(4):
            self.kf.H[i, i] = 1.0

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # high uncertainty on initial velocity
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = bbox_to_z(bbox)

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.confirmed = False

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox_to_z(bbox))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return z_to_bbox(self.kf.x[:4].reshape(-1))

    def get_state(self):
        return z_to_bbox(self.kf.x[:4].reshape(-1))


def associate_detections_to_trackers(detections, tracked_boxes, iou_threshold):
    if len(tracked_boxes) == 0:
        return [], list(range(len(detections))), []

    iou_matrix = iou_batch(detections, tracked_boxes)

    if min(iou_matrix.shape) > 0:
        row_idx, col_idx = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_idx, col_idx)))
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(tracked_boxes)) if t not in matched_indices[:, 1]]

    matches = []
    for d, t in matched_indices:
        if iou_matrix[d, t] < iou_threshold:
            unmatched_detections.append(d)
            unmatched_trackers.append(t)
        else:
            matches.append((d, t))

    return matches, unmatched_detections, unmatched_trackers


class Sort:
    """Multi-object tracker: feed it detections per frame, get back track IDs."""

    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        detections: np.ndarray of shape (N, 5) -> [x1,y1,x2,y2,conf], or (0,5) if none.
        Returns: np.ndarray of shape (M, 5) -> [x1,y1,x2,y2,track_id].
        """
        if detections is None or len(detections) == 0:
            detections = np.empty((0, 5))

        self.frame_count += 1

        predicted_boxes = []
        stale = []
        for i, trk in enumerate(self.trackers):
            box = trk.predict()
            if np.any(np.isnan(box)):
                stale.append(i)
            else:
                predicted_boxes.append(box)
        for i in reversed(stale):
            self.trackers.pop(i)

        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections[:, :4], predicted_boxes, self.iou_threshold
        )

        for d, t in matches:
            self.trackers[t].update(detections[d, :4])

        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d, :4]))

        results = []
        alive = []
        for trk in self.trackers:
            if trk.time_since_update < self.max_age:
                alive.append(trk)
                if not trk.confirmed and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                    trk.confirmed = True
                # Once confirmed, keep reporting (on Kalman-predicted position if
                # needed) through brief gaps -- e.g. a momentary occlusion where two
                # people's boxes merge into one detection for a frame or two -- rather
                # than requiring min_hits fresh re-matches, which reads to zone logic
                # as a real exit+re-entry for an object that never actually left.
                if trk.confirmed:
                    state = trk.get_state()
                    results.append(np.append(state, trk.id))
        self.trackers = alive

        return np.array(results) if results else np.empty((0, 5))
