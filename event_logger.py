"""Logs zone enter/exit events to CSV and keeps a rolling in-memory feed for display."""
import csv
import os
from collections import deque

import config


class EventLogger:
    def __init__(self, csv_path=config.DEFAULT_LOG_CSV,
                 max_display_lines=config.EVENT_OVERLAY_MAX_LINES):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        is_new_file = not os.path.exists(csv_path)
        self._file = open(csv_path, "a", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=["timestamp", "object_id", "zone", "event"]
        )
        if is_new_file:
            self._writer.writeheader()
            self._file.flush()

        self.recent_events = deque(maxlen=max_display_lines)

    def log(self, event):
        self._writer.writerow(event)
        self._file.flush()
        self.recent_events.append(event)
        print(f"[{event['timestamp']}] object {event['object_id']} "
              f"{event['event'].upper()} zone '{event['zone']}'")

    def log_many(self, events):
        for event in events:
            self.log(event)

    def close(self):
        self._file.close()
