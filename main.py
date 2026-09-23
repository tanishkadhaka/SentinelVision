"""
ZoneGuard end-to-end pipeline: detect -> track -> zone intrusion -> log.

    python main.py --video path/to/video.mp4 --zones zones.json

Press 'q' to quit the display window.
"""
import argparse
import csv
import os

import cv2
import numpy as np

import config
from detector import Detector
from event_logger import EventLogger
from sort_tracker import Sort, iou_batch
from zone_logic import ZoneManager, load_zones


def class_name_for(cls_id):
    return config.TARGET_CLASSES.get(cls_id, f"class_{cls_id}")


def open_debug_log(path):
    """CSV sink for --debug: one row per detection, new/lost track, and in-zone observation."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    f = open(path, "w", newline="")  # fresh file per debug run
    writer = csv.DictWriter(f, fieldnames=[
        "frame", "kind", "track_id", "class", "confidence", "zone", "x1", "y1", "x2", "y2"
    ])
    writer.writeheader()
    f.flush()
    return f, writer


def best_match_class(box_xyxy, raw_boxes, raw_classes):
    """
    Best-effort attribution of a tracked box back to the raw detection that most
    overlaps it this frame, purely for debug display (does not affect tracking).
    Returns (class_name, confidence) or (None, None) if no detection this frame
    overlaps at all -- meaning the track survived on Kalman prediction alone.
    """
    if raw_boxes is None or len(raw_boxes) == 0:
        return None, None
    ious = iou_batch(np.array([box_xyxy]), raw_boxes[:, :4])[0]
    best = int(np.argmax(ious))
    if ious[best] <= 0:
        return None, None
    return class_name_for(raw_classes[best]), float(raw_boxes[best, 4])


def draw_frame(frame, tracked_objects, zone_manager, recent_events):
    display = frame.copy()

    for zone in zone_manager.zones:
        cv2.polylines(display, [zone.points], True, config.ZONE_COLOR, 2)
        cv2.putText(display, zone.name, tuple(zone.points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.ZONE_COLOR, 2)

    for track_id, x1, y1, x2, y2 in tracked_objects:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        inside_any = bool(zone_manager.zones_containing(cx, cy))
        color = config.ALERT_COLOR if inside_any else config.BOX_COLOR

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, f"ID {int(track_id)}", (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(display, (cx, cy), 3, color, -1)

    for i, event in enumerate(reversed(recent_events)):
        text = f"{event['timestamp']} | ID {event['object_id']} {event['event'].upper()} {event['zone']}"
        cv2.putText(display, text, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return display


def main():
    parser = argparse.ArgumentParser(description="Run ZoneGuard intrusion detection on a video.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--zones", default=config.DEFAULT_ZONES_FILE, help="Path to zones.json")
    parser.add_argument("--weights", default=config.MODEL_WEIGHTS, help="YOLO weights file.")
    parser.add_argument("--log", default=config.DEFAULT_LOG_CSV, help="CSV path for event log.")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no GUI window).")
    parser.add_argument("--output", default="output.mp4",
                         help="Path to save the annotated output video (default: output.mp4).")
    parser.add_argument("--debug", action="store_true",
                         help="Log per-frame detection confidences, tracker config, "
                              "new/lost track events, and class labels for anything "
                              "inside a zone. Written to logs/debug.csv and console.")
    args = parser.parse_args()

    zones = load_zones(args.zones)
    if not zones:
        print(f"Warning: no zones loaded from {args.zones}. "
              f"Run zone_drawer.py first to define zones.")
    zone_manager = ZoneManager(zones)

    detector = Detector(weights=args.weights)
    tracker = Sort(
        max_age=config.SORT_MAX_AGE,
        min_hits=config.SORT_MIN_HITS,
        iou_threshold=config.SORT_IOU_THRESHOLD,
    )
    logger = EventLogger(csv_path=args.log)

    debug_file, debug_writer = (None, None)
    if args.debug:
        print(f"[DEBUG] Detector: weights={args.weights} "
              f"conf_threshold={config.CONF_THRESHOLD} nms_iou_threshold={config.IOU_THRESHOLD} "
              f"target_classes={config.TARGET_CLASSES}")
        print(f"[DEBUG] SORT tracker: max_age={config.SORT_MAX_AGE} "
              f"min_hits={config.SORT_MIN_HITS} iou_threshold={config.SORT_IOU_THRESHOLD}")
        debug_file, debug_writer = open_debug_log("logs/debug.csv")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {args.output}")
    print(f"Saving annotated output to {args.output}")

    window_name = "ZoneGuard"
    frame_idx = 0
    previous_track_ids = set()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            detections, det_classes = detector.detect_with_classes(frame)

            if args.debug:
                for box, cls_id in zip(detections, det_classes):
                    cname = class_name_for(cls_id)
                    print(f"[DEBUG] frame {frame_idx}: detection class={cname} "
                          f"conf={box[4]:.3f} box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")
                    debug_writer.writerow({
                        "frame": frame_idx, "kind": "detection", "track_id": "",
                        "class": cname, "confidence": f"{box[4]:.3f}", "zone": "",
                        "x1": f"{box[0]:.0f}", "y1": f"{box[1]:.0f}",
                        "x2": f"{box[2]:.0f}", "y2": f"{box[3]:.0f}",
                    })

            tracked = tracker.update(detections)  # [x1,y1,x2,y2,track_id] rows
            tracked_objects = [(row[4], row[0], row[1], row[2], row[3]) for row in tracked]

            if args.debug:
                current_ids = {int(row[4]) for row in tracked}
                for tid in current_ids - previous_track_ids:
                    row = next(r for r in tracked if int(r[4]) == tid)
                    cname, conf = best_match_class(row[:4], detections, det_classes)
                    cname = cname or "no-matching-detection-this-frame"
                    conf_str = f"{conf:.3f}" if conf is not None else ""
                    print(f"[DEBUG] frame {frame_idx}: NEW TRACK id={tid} class={cname} conf={conf_str}")
                    debug_writer.writerow({
                        "frame": frame_idx, "kind": "new_track", "track_id": tid,
                        "class": cname, "confidence": conf_str, "zone": "",
                        "x1": f"{row[0]:.0f}", "y1": f"{row[1]:.0f}",
                        "x2": f"{row[2]:.0f}", "y2": f"{row[3]:.0f}",
                    })
                for tid in previous_track_ids - current_ids:
                    print(f"[DEBUG] frame {frame_idx}: LOST TRACK id={tid}")
                    debug_writer.writerow({
                        "frame": frame_idx, "kind": "lost_track", "track_id": tid,
                        "class": "", "confidence": "", "zone": "",
                        "x1": "", "y1": "", "x2": "", "y2": "",
                    })
                previous_track_ids = current_ids

                for track_id, x1, y1, x2, y2 in tracked_objects:
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    for zname in zone_manager.zones_containing(cx, cy):
                        cname, conf = best_match_class(
                            np.array([x1, y1, x2, y2]), detections, det_classes
                        )
                        cname = cname or "no-matching-detection-this-frame(prediction-only)"
                        conf_str = f"{conf:.3f}" if conf is not None else ""
                        print(f"[DEBUG] frame {frame_idx}: track {int(track_id)} inside zone "
                              f"'{zname}' class={cname} conf={conf_str}")
                        debug_writer.writerow({
                            "frame": frame_idx, "kind": "in_zone", "track_id": int(track_id),
                            "class": cname, "confidence": conf_str, "zone": zname,
                            "x1": f"{x1:.0f}", "y1": f"{y1:.0f}",
                            "x2": f"{x2:.0f}", "y2": f"{y2:.0f}",
                        })

            events = zone_manager.update(tracked_objects)
            logger.log_many(events)

            display = draw_frame(frame, tracked_objects, zone_manager, logger.recent_events)
            out.write(display)

            if not args.no_display:
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.close()
        if debug_file:
            debug_file.close()


if __name__ == "__main__":
    main()