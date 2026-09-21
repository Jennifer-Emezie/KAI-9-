#!/usr/bin/env python3
"""
ai_detect.py  —  Pi 4 + IMX500 AI Camera — person detection only

Filters the IMX500's object detection output down to just "person",
reporting confidence and bounding box for each person currently in frame.

Usage:
    python ai_detect.py
    python ai_detect.py --model imx500-models/imx500_network_yolo11n_pp.rpk
    python ai_detect.py --threshold 0.4

Usage as a module:
    from ai_detect import CameraDetector
    detector = CameraDetector()
    for snapshot in detector.stream():
        for person in snapshot["detections"]:
            print(person["confidence"], person["box"])
"""

import argparse
import os
import threading
import time

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500


# ---------------------------------------------------------------------------
# Fallback COCO-80 labels (only needed to look up "person"'s class index)
# ---------------------------------------------------------------------------

COCO_LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ---------------------------------------------------------------------------
# Stance estimation (bounding box aspect ratio heuristic)
# ---------------------------------------------------------------------------

# width / height ratio thresholds — tune these for your camera's mounting
# angle and height. A camera mounted high and angled down will see a
# standing person as relatively "wider" than one mounted at eye level.
STANCE_STANDING_RATIO = 0.5   # ratio below this  -> standing
STANCE_LYING_RATIO    = 1.5   # ratio at/above this -> lying down
                               # in between        -> sitting/crouching

def estimate_stance(box: dict) -> str:
    """
    Rough heuristic for a person's stance from their bounding box dimensions.

        ratio = box["w"] / box["h"]

        ratio < STANCE_STANDING_RATIO  -> "standing"  (tall, narrow box)
        ratio < STANCE_LYING_RATIO     -> "sitting"   (roughly square box)
        otherwise                      -> "lying_down" (wide, short box)

    This is a crude signal, not a substitute for proper pose estimation —
    it can be thrown off by partial occlusion, the person being near the
    edge of frame, or unusual camera angles. It's useful as a cheap
    first-pass "something's wrong" trigger (e.g. lying_down + not moving
    for a while), but treat it as a hint rather than ground truth.

    Returns one of: "standing", "sitting", "lying_down", "unknown"
    (the latter if the box has zero height, which shouldn't normally happen).
    """
    w = box.get("w", 0)
    h = box.get("h", 0)

    if h <= 0:
        return "unknown"

    ratio = w / h

    if ratio < STANCE_STANDING_RATIO:
        return "standing"
    elif ratio < STANCE_LYING_RATIO:
        return "sitting"
    else:
        return "lying_down"


# ---------------------------------------------------------------------------
# Parsers — only return "person" detections
# ---------------------------------------------------------------------------

def parse_yolo(outputs, frame_w, frame_h, threshold, person_class_id):
    scale_x = frame_w / 640
    scale_y = frame_h / 640
    results = []
    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    for i in range(len(scores)):
        if int(classes[i]) != person_class_id:
            continue
        score = float(scores[i])
        if score < threshold:
            continue
        cx_n, cy_n, w_n, h_n = boxes[i]
        bw = int(w_n * scale_x)
        bh = int(h_n * scale_y)
        x1 = int(cx_n * scale_x) - bw // 2
        y1 = int(cy_n * scale_y) - bh // 2
        box = {"x": x1, "y": y1, "w": bw, "h": bh}
        results.append({
            "label":      "person",
            "confidence": round(score, 3),
            "box":        box,
            "stance":     estimate_stance(box),
        })
    return results


def parse_efficientdet(outputs, frame_w, frame_h, threshold, person_class_id):
    scale_x = frame_w / 320
    scale_y = frame_h / 320
    results = []
    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    count = int(outputs[3].item())
    for i in range(count):
        class_id = max(0, int(classes[i]) - 1)   # EfficientDet is 1-indexed
        if class_id != person_class_id:
            continue
        score = float(scores[i])
        if score < threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        x1 = int(x_min * scale_x)
        y1 = int(y_min * scale_y)
        bw = int((x_max - x_min) * scale_x)
        bh = int((y_max - y_min) * scale_y)
        box = {"x": x1, "y": y1, "w": bw, "h": bh}
        results.append({
            "label":      "person",
            "confidence": round(score, 3),
            "box":        box,
            "stance":     estimate_stance(box),
        })
    return results


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CameraDetector:
    """
    Streams snapshots every `interval` seconds.  Each snapshot is a dict:

        {
            "timestamp":  "14:32:01",
            "detections": [
                {"label": "person", "confidence": 0.91, "box": {"x": 120, "y": 45, "w": 80, "h": 200}},
                {"label": "person", "confidence": 0.76, "box": {"x": 300, "y": 110, "w": 60, "h": 90}},
            ]
        }

    "detections" only ever contains people — every other COCO class is
    filtered out before it reaches you. An empty list means no person is
    currently in frame.
    """

    def __init__(
        self,
        model_path: str  = "imx500-models/imx500_network_yolo11n_pp.rpk",
        labels_path: str = "imx500-models/coco_labels.txt",
        threshold: float = 0.5,
        fps: float       = 30.0,
        interval: float  = 0.5,
        resolution: tuple = (1920, 1080),
    ):
        self.threshold = threshold
        self.interval  = interval

        # Labels — only needed to find "person"'s class index
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                labels = [l.strip() for l in f if l.strip()]
        else:
            print(f"[WARN] Labels file '{labels_path}' not found — using built-in COCO-80.")
            labels = COCO_LABELS

        if "person" not in labels:
            raise ValueError("'person' not found in labels file — check your model/labels.")
        self.person_class_id = labels.index("person")

        # Parser
        model_name = os.path.basename(model_path)
        if "yolo" in model_name.lower():
            self._parse = parse_yolo
            print("[INFO] Parser: YOLO")
        else:
            self._parse = parse_efficientdet
            print("[INFO] Parser: EfficientDet")

        # Camera
        print(f"[INFO] Loading model: {model_name}")
        self._imx500 = IMX500(model_path)
        self._picam2 = Picamera2(self._imx500.camera_num)
        self._picam2.start(
            self._picam2.create_preview_configuration(
                main={"size": resolution},
                controls={"FrameRate": fps},
                buffer_count=12,
            )
        )

        # Block until firmware is ready (can take up to ~2 min on first boot)
        print("[INFO] Waiting for IMX500 firmware to load (may take up to 2 min)…")
        ready = threading.Event()

        def _wait():
            self._picam2.capture_metadata()
            ready.set()

        t = threading.Thread(target=_wait, daemon=True)
        t.start()
        t.join(timeout=180)
        if not ready.is_set():
            self._picam2.stop()
            self._picam2.close()
            raise RuntimeError("IMX500 did not produce a frame within 3 minutes.")

        cfg = self._picam2.camera_configuration()
        self.frame_w = cfg["main"]["size"][0]
        self.frame_h = cfg["main"]["size"][1]
        print(f"[INFO] Ready — {self.frame_w}×{self.frame_h} @ {fps:.0f} FPS\n")

    def _latest_detections(self) -> list:
        """
        Drain frames at full camera speed until one has AI outputs, then
        return its person detections (which may be an empty list if nobody
        is in frame — that's a valid result, not an error). The IMX500 only
        attaches inference results to every Nth frame; calling
        capture_metadata() in a tight loop and discarding None outputs means
        we never confuse "no AI output yet" with "no person in frame". A
        5-second watchdog stops the loop if the camera stalls.
        """
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            result = [None]

            def _fetch():
                try:
                    result[0] = self._picam2.capture_metadata()
                except Exception:
                    pass

            t = threading.Thread(target=_fetch, daemon=True)
            t.start()
            t.join(timeout=5.0)

            metadata = result[0]
            if metadata is None:
                continue

            outputs = self._imx500.get_outputs(metadata, add_batch=True)
            if outputs is None:
                continue     # this frame had no AI output — try the next one

            return self._parse(
                outputs, self.frame_w, self.frame_h, self.threshold, self.person_class_id
            )

        print("[WARN] No AI output received within 5 s — returning empty frame")
        return []

    def get_snapshot(self) -> dict:
        """Return the next snapshot that contains real AI output."""
        return {
            "timestamp":  time.strftime("%H:%M:%S"),
            "detections": self._latest_detections(),
        }

    def close(self):
        """
        Fully release the camera hardware so a future CameraDetector can
        acquire() it again. `Picamera2.stop()` alone only halts the
        stream — the camera stays "Configured" at the libcamera level and
        a later instance's acquire() will fail. `close()` releases it
        back to "Available". Safe to call more than once.
        """
        try:
            self._picam2.stop()
        except Exception:
            pass
        try:
            self._picam2.close()
        except Exception:
            pass

    def stream(self):
        """
        Generator — yields one snapshot every `interval` seconds, guaranteed
        to contain the most recent real AI output (never an empty frame caused
        by the IMX500 skipping inference on intermediate frames).
        Stops cleanly on Ctrl+C.
        """
        try:
            while True:
                start = time.monotonic()
                yield self.get_snapshot()
                elapsed = time.monotonic() - start
                remaining = self.interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
            print("\n[INFO] Camera stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="IMX500 person detector")
    p.add_argument("--model",     default="imx500-models/imx500_network_yolo11n_pp.rpk")
    p.add_argument("--labels",    default="imx500-models/coco_labels.txt")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Minimum confidence (0–1)")
    p.add_argument("--fps",       type=float, default=30.0)
    p.add_argument("--interval",  type=float, default=0.5,
                   help="Seconds between checks")
    p.add_argument("--resolution", default="1920x1080",
                   help="Main stream resolution as WIDTHxHEIGHT (default: 1920x1080)")
    return p.parse_args()


def main():
    args = parse_args()
    width, height = (int(v) for v in args.resolution.lower().split("x"))
    detector = CameraDetector(
        model_path=args.model,
        labels_path=args.labels,
        threshold=args.threshold,
        fps=args.fps,
        interval=args.interval,
        resolution=(width, height),
    )

    for snapshot in detector.stream():
        for person in snapshot["detections"]:
            b = person["box"]
            print(
                f"[{snapshot['timestamp']}] person  "
                f"conf={person['confidence']:.0%}  "
                f"stance={person['stance']:<10}  "
                f"box=(x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']})"
            )


if __name__ == "__main__":
    main()
