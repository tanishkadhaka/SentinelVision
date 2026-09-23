"""
Interactive polygon zone editor.

Run standalone against a video file to draw zones on its first frame and
save them to zones.json:

    python zone_drawer.py --video path/to/video.mp4 --zones zones.json

Controls:
    Left click   - add a point to the polygon currently being drawn
    Right click  - undo the last point
    Enter / c    - close the current polygon and name it (typed in console)
    s            - save all zones to the zones file
    r            - reset (delete all zones)
    q / Esc      - quit
"""
import argparse

import cv2

import config
from zone_logic import Zone, load_zones, save_zones


class ZoneDrawer:
    def __init__(self, frame, zones_path):
        self.frame = frame
        self.zones_path = zones_path
        self.zones = load_zones(zones_path)
        self.current_points = []
        self.window_name = "ZoneGuard - Draw Zones"

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        print("Zone Drawer controls:")
        print("  left click   = add point")
        print("  right click  = undo last point")
        print("  enter / c    = close polygon and name it")
        print("  s            = save zones.json")
        print("  r            = reset all zones")
        print("  q / esc      = quit")

        while True:
            display = self._render()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                break
            elif key in (13, ord("c")):
                self._close_polygon()
            elif key == ord("s"):
                save_zones(self.zones_path, self.zones)
                print(f"Saved {len(self.zones)} zone(s) to {self.zones_path}")
            elif key == ord("r"):
                self.zones = []
                self.current_points = []
                print("All zones cleared (not yet saved).")

        cv2.destroyWindow(self.window_name)
        return self.zones

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.current_points:
            self.current_points.pop()

    def _close_polygon(self):
        if len(self.current_points) < 3:
            print("Need at least 3 points to close a polygon.")
            return
        name = input("Zone name: ").strip() or f"zone_{len(self.zones) + 1}"
        self.zones.append(Zone(name, self.current_points))
        print(f"Added zone '{name}' with {len(self.current_points)} points.")
        self.current_points = []

    def _render(self):
        overlay = self.frame.copy()

        for zone in self.zones:
            cv2.fillPoly(overlay, [zone.points], config.ZONE_COLOR)
        display = cv2.addWeighted(
            overlay, config.ZONE_FILL_ALPHA, self.frame, 1 - config.ZONE_FILL_ALPHA, 0
        )

        for zone in self.zones:
            cv2.polylines(display, [zone.points], True, config.ZONE_COLOR, 2)
            label_pos = tuple(zone.points[0])
            cv2.putText(display, zone.name, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.ZONE_COLOR, 2)

        if self.current_points:
            for pt in self.current_points:
                cv2.circle(display, pt, 4, (255, 255, 255), -1)
            for i in range(len(self.current_points) - 1):
                cv2.line(display, self.current_points[i], self.current_points[i + 1],
                          (255, 255, 255), 2)

        return display


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from video: {video_path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description="Draw polygon zones on a video's first frame.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--zones", default=config.DEFAULT_ZONES_FILE, help="Path to zones.json")
    args = parser.parse_args()

    frame = get_first_frame(args.video)
    drawer = ZoneDrawer(frame, args.zones)
    zones = drawer.run()
    save_zones(args.zones, zones)
    print(f"Final save: {len(zones)} zone(s) written to {args.zones}")


if __name__ == "__main__":
    main()
