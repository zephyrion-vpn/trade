"""One-shot detector for 'Купить' buttons on the right side of the screen.

What it does:
- Waits 3 seconds after launch.
- Captures the primary screen once.
- Detects blue-colored regions (potential buy buttons) by HSV color mask.
- Optionally compares detected regions with a template image (buy_template.png).
- Draws a red highlight over the detected button areas.
- Saves the annotated screenshot and opens a preview window.

Install:
    pip install opencv-python mss numpy pillow

No Tesseract OCR needed!

Run:
    python trade.py

Optional:
    Place a small screenshot of your "Купить" button as 'buy_template.png' 
    in the current folder for template matching.

Output:
- annotated_buy_button.png in the current folder.
- ocr_debug_right.png if debugging is enabled.
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
    debug_output_path: str = "ocr_debug_right.png"
    template_path: str = "buy_template.png"  # Template image for matching

    # Only search the right part of the screen, where the buy buttons are.
    right_crop_ratio: float = 0.55

    # Visual highlight.
    overlay_alpha: float = 0.28

    # ── Blue-button color detection (HSV) ──────────────────────────────────
    use_color_detection: bool = True
    # HSV hue range for "game blue" buttons. Adjust if your button colour
    # looks more cyan (lower hue) or violet (higher hue).
    blue_hsv_lower: Tuple[int, int, int] = (90, 50, 50)
    blue_hsv_upper: Tuple[int, int, int] = (145, 255, 255)
    # Minimum pixel area of a blue contour to be considered a button candidate.
    blue_min_area: int = 400
    # Acceptable aspect-ratio (width/height) range for button-like shapes.
    blue_aspect_min: float = 1.2
    blue_aspect_max: float = 10.0
    # Extra padding added around each detected blue region before OCR.
    blue_region_pad: int = 6

    # ── Template matching ───────────────────────────────────────────────────
    # Threshold for template matching (0.0 to 1.0, higher = stricter match)
    template_threshold: float = 0.75
    # Non-maximum suppression threshold for overlapping template matches
    template_nms_threshold: float = 0.5

    # Debugging.
    save_debug_image: bool = True
    print_ocr_debug: bool = True


cfg = Config()


def grab_primary_screen() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def load_template() -> Optional[np.ndarray]:
    """Load template image if it exists."""
    template_path = Path(cfg.template_path)
    if template_path.exists():
        template = cv2.imread(str(template_path))
        if template is not None:
            print(f"Loaded template: {template_path} ({template.shape[1]}x{template.shape[0]})")
            return template
    return None


# ── Blue-color region detection ──────────────────────────────────────────

def find_blue_regions(frame: np.ndarray, x0: int) -> List[Tuple[int, int, int, int]]:
    """Return (x, y, w, h) rectangles (in full-frame coords) of blue UI regions.

    Searches only the right portion of the frame (x >= x0) to match the
    same area as the OCR pass.  Each returned box is padded slightly so
    that OCR has a little context around the button text.
    """
    crop = frame[:, x0:]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower = np.array(cfg.blue_hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.blue_hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Close small gaps inside buttons, remove speckle noise.
    k_close = np.ones((7, 7), np.uint8)
    k_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < cfg.blue_min_area:
            continue
        aspect = cw / max(ch, 1)
        if not (cfg.blue_aspect_min <= aspect <= cfg.blue_aspect_max):
            continue

        # Translate back to full-frame coordinates and add padding.
        pad = cfg.blue_region_pad
        rx = max(0, cx + x0 - pad)
        ry = max(0, cy - pad)
        rw = cw + pad * 2
        rh = ch + pad * 2
        regions.append((rx, ry, rw, rh))

    if cfg.print_ocr_debug:
        print(f"  [color] found {len(regions)} blue region(s) in right crop")

    return regions


def find_by_template(frame: np.ndarray, template: np.ndarray, x0: int) -> List[Tuple[int, int, int, int, float]]:
    """Find button locations using template matching.
    
    Returns list of (x, y, w, h, confidence) tuples.
    """
    crop = frame[:, x0:]
    
    # Match template against the crop
    result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]
    
    # Find all matches above threshold
    locations = np.where(result >= cfg.template_threshold)
    
    matches: List[Tuple[int, int, int, int, float]] = []
    for pt in zip(*locations[::-1]):
        conf = float(result[pt[1], pt[0]])
        x = pt[0] + x0  # Translate to full-frame coordinates
        y = pt[1]
        matches.append((x, y, w, h, conf))
    
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
    
    if cfg.print_ocr_debug:
        print(f"  [template] found {len(filtered)} match(es) with threshold={cfg.template_threshold}")
    
    return filtered


def find_by_color_with_confidence(frame: np.ndarray, x0: int) -> List[Tuple[int, int, int, int, float]]:
    """Find button locations using color detection, assign confidence based on color saturation.
    
    Returns list of (x, y, w, h, confidence) tuples.
    """
    regions = find_blue_regions(frame, x0)
    results: List[Tuple[int, int, int, int, float]] = []
    
    for rx, ry, rw, rh in regions:
        # Extract the region and calculate average saturation as confidence
        region = frame[ry:ry+rh, rx:rx+rw]
        if region.size == 0:
            continue
        
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv_region[:, :, 1]
        avg_saturation = float(np.mean(saturation)) / 255.0  # Normalize to 0-1
        
        # Confidence based on how "blue" the region is (saturation + hue consistency)
        confidence = min(1.0, avg_saturation * 1.2)  # Boost slightly
        
        results.append((rx, ry, rw, rh, confidence))
    
    return results


# ── main detection function ────────────────────────────────────────────────────

def find_buy_button_boxes(frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
    h, w = frame.shape[:2]
    x0 = int(w * cfg.right_crop_ratio)
    
    all_boxes: List[Tuple[int, int, int, int, float, str]] = []
    
    # Load template if available
    template = load_template()
    
    # Pass 1: Template matching (most reliable if template exists)
    if template is not None:
        template_matches = find_by_template(frame, template, x0)
        for x, y, bw, bh, conf in template_matches:
            all_boxes.append((x, y, bw, bh, conf * 100, "BUY"))
    
    # Pass 2: Color-based detection (works without template)
    if cfg.use_color_detection:
        color_matches = find_by_color_with_confidence(frame, x0)
        for x, y, bw, bh, conf in color_matches:
            # Only add if not already found by template matching
            is_duplicate = False
            for fx, fy, fw, fh, _, _ in all_boxes:
                ix1 = max(x, fx)
                iy1 = max(y, fy)
                ix2 = min(x + bw, fx + fw)
                iy2 = min(y + bh, fy + fh)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area = bw * bh
                farea = fw * fh
                if inter / float(min(area, farea) + 1e-6) > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_boxes.append((x, y, bw, bh, conf * 100, "BUY"))
    
    if cfg.print_ocr_debug:
        print(f"Right crop starts at x={x0}, frame size={w}x{h}")
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


def save_debug_image(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    x0 = int(w * cfg.right_crop_ratio)
    crop = frame[:, x0:]

    # Save a blue-mask visualisation so it's easy to check colour tuning.
    if cfg.use_color_detection:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower = np.array(cfg.blue_hsv_lower, dtype=np.uint8)
        upper = np.array(cfg.blue_hsv_upper, dtype=np.uint8)
        blue_mask = cv2.inRange(hsv, lower, upper)
        blue_debug_path = cfg.debug_output_path.replace(".png", "_blue_mask.png")
        cv2.imwrite(blue_debug_path, blue_mask)
        print(f"Saved blue-mask debug image: {Path(blue_debug_path).resolve()}")


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
    if cfg.save_debug_image:
        try:
            save_debug_image(frame)
        except Exception as e:
            print(f"Could not save debug image: {e}")

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