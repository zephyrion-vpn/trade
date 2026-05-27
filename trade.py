"""One-shot detector for buttons using template matching.

What it does:
- Waits 3 seconds after launch.
- Captures the primary screen once.
- Finds buttons by comparing with a template screenshot (button_template.png).
- Draws a red highlight over the detected button areas.
- Saves the annotated screenshot and opens a preview window.

Install:
    pip install opencv-python mss numpy pillow

No Tesseract OCR needed! No color detection! Pure template matching!

Run:
    python trade.py

Required:
    Place a screenshot of your button as 'button_template.png' 
    in the current folder for template matching.

Output:
- annotated_buy_button.png in the current folder.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import mss
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


@dataclass
class Config:
    delay_seconds: int = 3
    output_path: str = "annotated_buy_button.png"
    template_path: str = "button_template.png"  # Template image for matching

    # Visual highlight.
    overlay_alpha: float = 0.28

    # ── Template matching ───────────────────────────────────────────────────
    # Threshold for template matching (0.0 to 1.0, higher = stricter match)
    template_threshold: float = 0.75
    # Non-maximum suppression threshold for overlapping template matches
    template_nms_threshold: float = 0.5

    # Debugging.
    print_debug: bool = True


cfg = Config()


def grab_primary_screen() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def load_template() -> np.ndarray:
    """Load template image. Required for detection."""
    template_path = Path(cfg.template_path)
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template file '{cfg.template_path}' not found!\n"
            f"Please place a screenshot of your button as '{cfg.template_path}' in the current folder."
        )
    template = cv2.imread(str(template_path))
    if template is None:
        raise ValueError(f"Could not load template from '{template_path}'")
    print(f"Loaded template: {template_path} ({template.shape[1]}x{template.shape[0]})")
    return template


def find_by_template(frame: np.ndarray, template: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """Find button locations using template matching.
    
    Returns list of (x, y, w, h, confidence) tuples.
    """
    # Match template against the full frame
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape[:2]
    
    # Find all matches above threshold
    locations = np.where(result >= cfg.template_threshold)
    
    matches: List[Tuple[int, int, int, int, float]] = []
    for pt in zip(*locations[::-1]):
        conf = float(result[pt[1], pt[0]])
        x, y = pt[0], pt[1]
        matches.append((x, y, tw, th, conf))
    
    # Non-maximum suppression: remove overlapping detections
    filtered: List[Tuple[int, int, int, int, float]] = []
    for x, y, bw, bh, conf in sorted(matches, key=lambda m: m[4], reverse=True):
        keep = True
        for fx, fy, fw, fh, _ in filtered:
            # Calculate IoU
            ix1 = max(x, fx)
            iy1 = max(y, fy)
            ix2 = min(x + bw, fx + fw)
            iy2 = min(y + bh, fy + fh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = bw * bh
            farea = fw * fh
            if inter / float(min(area, farea) + 1e-6) > cfg.template_nms_threshold:
                keep = False
                break
        if keep:
            filtered.append((x, y, bw, bh, conf))
    
    if cfg.print_debug:
        print(f"  Found {len(filtered)} match(es) with threshold={cfg.template_threshold}")
    
    return filtered


# ── main detection function ────────────────────────────────────────────────────

def find_buy_button_boxes(frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
    template = load_template()
    
    # Find buttons using template matching
    matches = find_by_template(frame, template)
    
    all_boxes: List[Tuple[int, int, int, int, float, str]] = []
    for x, y, bw, bh, conf in matches:
        all_boxes.append((x, y, bw, bh, conf * 100, "BUY"))
    
    if cfg.print_debug:
        h, w = frame.shape[:2]
        print(f"Frame size={w}x{h}")
        total = len(all_boxes)
        print(f"  total candidates: {total}")
        for idx, (x, y, bw, bh, conf, text) in enumerate(all_boxes[:20], 1):
            print(f"  candidate {idx}: text={text!r}, conf={conf:.1f}, box=({x},{y},{bw},{bh})")
    
    return all_boxes


def annotate_frame(frame: np.ndarray, boxes: List[Tuple[int, int, int, int, float, str]]) -> np.ndarray:
    result = frame.copy()
    overlay = frame.copy()

    for x, y, w, h, conf, text in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), thickness=-1)

    if boxes:
        result = cv2.addWeighted(overlay, cfg.overlay_alpha, result, 1 - cfg.overlay_alpha, 0)
        for x, y, w, h, conf, text in boxes:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            label = f"BUY {conf:.0f}%"
            cv2.putText(
                result,
                label,
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            result,
            "Buy button not found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return result





def show_preview(image_bgr: np.ndarray) -> None:
    root = tk.Tk()
    root.title("Annotated screenshot")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tk_img = ImageTk.PhotoImage(image=image)

    label = tk.Label(root, image=tk_img)
    label.image = tk_img
    label.pack()

    info = tk.Label(root, text=f"Saved to: {Path(cfg.output_path).resolve()}")
    info.pack(pady=8)

    root.after(12000, root.destroy)
    root.mainloop()


def main() -> int:
    print(f"Starting in {cfg.delay_seconds} seconds...")
    time.sleep(cfg.delay_seconds)

    frame = grab_primary_screen()

    boxes = find_buy_button_boxes(frame)
    annotated = annotate_frame(frame, boxes)

    out_path = Path(cfg.output_path)
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved: {out_path.resolve()}")
    print(f"Detected buy buttons: {len(boxes)}")
    for i, (x, y, w, h, conf, text) in enumerate(boxes, 1):
        print(f"  {i}. text={text!r}, conf={conf:.1f}, box=({x},{y},{w},{h})")

    show_preview(annotated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())