"""Zone definitions + enter/exit detection for tracked objects."""
import json
import os
import time

import cv2
import numpy as np


class Zone:
    def __init__(self, name, points):
        self.name = name
        self.points = np.array(points, dtype=np.int32)

    def contains(self, x, y):
        """True if point (x, y) is inside or on the boundary of this zone."""
        result = cv2.pointPolygonTest(self.points, (float(x), float(y)), False)
        return result >= 0

    def to_dict(self):
        return {"name": self.name, "points": self.points.tolist()}

    @staticmethod
    def from_dict(d):
        return Zone(d["name"], d["points"])


def load_zones(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return [Zone.from_dict(z) for z in data]


def save_zones(path, zones):
    with open(path, "w") as f:
        json.dump([z.to_dict() for z in zones], f, indent=2)


class ZoneManager:
    """
    Tracks which zones each object ID currently occupies, and emits
    enter/exit events when that membership changes between frames.
    """

    def __init__(self, zones):
        self.zones = zones
        # track_id -> set of zone names the object was inside on the last frame
        self._object_zone_state = {}

    def update(self, tracked_objects):
        """
        tracked_objects: iterable of (track_id, x1, y1, x2, y2)
        Returns a list of event dicts: {timestamp, object_id, zone, event}
        """
        events = []
        seen_ids = set()

        for track_id, x1, y1, x2, y2 in tracked_objects:
            seen_ids.add(track_id)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            previously_inside = self._object_zone_state.get(track_id, set())
            currently_inside = set()

            for zone in self.zones:
                if zone.contains(cx, cy):
                    currently_inside.add(zone.name)

            entered = currently_inside - previously_inside
            exited = previously_inside - currently_inside

            for zone_name in entered:
                events.append(self._make_event(track_id, zone_name, "enter"))
            for zone_name in exited:
                events.append(self._make_event(track_id, zone_name, "exit"))

            self._object_zone_state[track_id] = currently_inside

        # Objects that disappeared this frame (track lost) count as exiting
        # any zone they were still inside.
        lost_ids = set(self._object_zone_state.keys()) - seen_ids
        for track_id in lost_ids:
            for zone_name in self._object_zone_state[track_id]:
                events.append(self._make_event(track_id, zone_name, "exit"))
            del self._object_zone_state[track_id]

        return events

    @staticmethod
    def _make_event(track_id, zone_name, event_type):
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "object_id": int(track_id),
            "zone": zone_name,
            "event": event_type,
        }

    def zones_containing(self, x, y):
        return [z.name for z in self.zones if z.contains(x, y)]
