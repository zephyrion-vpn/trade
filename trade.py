"""One-shot OCR annotator for 'Купить' buttons on the right side of the screen.

What it does:
- Waits 3 seconds after launch.
- Captures the primary screen once.
- Detects blue-colored regions (potential buy buttons) by HSV color mask.
- Uses OCR to find text like 'Купить' on the right side and in detected blue regions.
- Expands the OCR text box into a likely button rectangle.
- Draws a red highlight over the detected button areas.
- Saves the annotated screenshot and opens a preview window.

Install:
    pip install opencv-python mss numpy pillow pytesseract

You also need Tesseract OCR installed, with Russian language data available.

Run:
    python trade.py

Output:
- annotated_buy_button.png in the current folder.
- ocr_debug_right.png if debugging is enabled.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# ВАЖНО: Установить TESSDATA_PREFIX ДО импорта pytesseract!
# Это критично для работы с путями, содержащими кириллицу на Windows.
# =============================================================================

def get_tessdata_dir_early() -> Path:
    """Раннее получение tessdata директории (до импорта pytesseract)."""
    # Проверяем переменную окружения TESSDATA_PREFIX (пользовательский путь)
    tessdata_prefix = os.environ.get("TESSDATA_PREFIX", "").strip()
    if tessdata_prefix:
        candidate = Path(tessdata_prefix)
        if candidate.exists():
            return candidate
    
    # Проверяем стандартный путь C:\tesseract_data
    custom_path = Path(r"C:\tesseract_data")
    if custom_path.exists():
        return custom_path
    
    # Авто-определение пути
    tesseract_cmd = os.environ.get("TESSERACT_PATH", "")
    
    if not tesseract_cmd:
        import shutil
        tesseract_cmd = shutil.which("tesseract") or ""
    
    if not tesseract_cmd:
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        ]
        for path in possible_paths:
            if path and Path(path).exists():
                tesseract_cmd = path
                break
    
    if tesseract_cmd:
        exe_dir = Path(tesseract_cmd).resolve().parent
        tessdata_dir = exe_dir / "tessdata"
        if tessdata_dir.exists():
            return tessdata_dir
    
    # Fallback
    return Path(".")


# Устанавливаем TESSDATA_PREFIX до импорта pytesseract
_tessdata_early = get_tessdata_dir_early()
if _tessdata_early.exists():
    # Используем краткий путь без кириллицы, если задан TESSDATA_PREFIX
    user_prefix = os.environ.get("TESSDATA_PREFIX", "").strip()
    if user_prefix:
        os.environ["TESSDATA_PREFIX"] = user_prefix
    else:
        os.environ["TESSDATA_PREFIX"] = str(_tessdata_early)

import cv2
import mss
import numpy as np
import pytesseract
import pytesseract.pytesseract as pt
from PIL import Image, ImageTk
import tkinter as tk


# Tesseract executable path - try to use from PATH first, otherwise check common locations
TESSERACT_CMD = os.environ.get("TESSERACT_PATH", "")

if not TESSERACT_CMD:
    # Try to find tesseract in standard locations
    import shutil
    TESSERACT_CMD = shutil.which("tesseract") or ""
    
if not TESSERACT_CMD:
    # Fallback to common Windows installation paths
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
    ]
    for path in possible_paths:
        if path and Path(path).exists():
            TESSERACT_CMD = path
            break

if not TESSERACT_CMD:
    raise RuntimeError(
        "Tesseract OCR не найден. Установите Tesseract и либо:\n"
        "1. Добавьте его в PATH\n"
        "2. Или задайте переменную окружения TESSERACT_PATH с полным путём к tesseract.exe"
    )


# Функция get_tessdata_dir() удалена - теперь используется get_tessdata_dir_early()
# которая вызывается до импорта pytesseract для корректной установки TESSDATA_PREFIX

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Fix Windows decoding issues in pytesseract when Tesseract returns
# localized error messages in cp1251/ANSI encoding.
if sys.platform.startswith("win"):
    try:
        pt.DEFAULT_ENCODING = "mbcs"
    except Exception:
        pt.DEFAULT_ENCODING = "cp1251"

# TESSDATA_DIR уже установлен на уровне модуля (до импорта pytesseract)
# Используем то же значение для согласованности
TESSDATA_DIR = Path(os.environ.get("TESSDATA_PREFIX", str(_tessdata_early)))


@dataclass
class Config:
    delay_seconds: int = 3
    output_path: str = "annotated_buy_button.png"
    debug_output_path: str = "ocr_debug_right.png"

    # Only search the right part of the screen, where the buy buttons are.
    right_crop_ratio: float = 0.55  # slightly wider than before (was 0.58)

    # OCR settings.
    ocr_lang: str = "rus+eng"
    # LOWERED from 20.0 → 5.0 so weak OCR hits are not discarded.
    min_confidence: float = 5.0

    # Visual highlight.
    overlay_alpha: float = 0.28

    # OCR preprocessing.
    upscale: int = 2
    # Extended PSM list: added modes 3 (auto) and 7 (single line) for better coverage.
    psm_modes: Tuple[int, ...] = (3, 6, 7, 11)

    # Search target.
    target_prefixes: Tuple[str, ...] = ("КУП",)

    # ── Blue-button color detection (HSV) ──────────────────────────────────
    # Enable the HSV-based blue region pre-scan that runs OCR on each
    # detected blue patch individually.  Usually much more reliable than
    # whole-screen OCR when the button stands out by colour.
    use_color_detection: bool = True
    # HSV hue range for "game blue" buttons.  Adjust if your button colour
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

    # Debugging.
    save_debug_image: bool = True
    print_ocr_debug: bool = True


cfg = Config()


def ensure_tesseract_ready() -> None:
    exe = Path(TESSERACT_CMD)
    if not exe.exists():
        raise FileNotFoundError(
            f"Не найден tesseract.exe по пути: {TESSERACT_CMD}\n"
            f"Проверь путь в переменной TESSERACT_CMD или установи Tesseract."
        )

    # TESSDATA_DIR уже получен через get_tessdata_dir_early(), который проверяет TESSDATA_PREFIX
    if not TESSDATA_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка tessdata: {TESSDATA_DIR}\n"
            f"Проверь установку Tesseract и пути к языковым файлам."
        )

    rus = TESSDATA_DIR / "rus.traineddata"
    eng = TESSDATA_DIR / "eng.traineddata"

    missing = []
    if not rus.exists():
        missing.append(str(rus))
    if not eng.exists():
        missing.append(str(eng))

    if missing:
        raise FileNotFoundError(
            "Не найдены языковые файлы Tesseract:\n"
            + "\n".join(missing)
            + "\nНужны как минимум rus.traineddata и eng.traineddata.\n\n"
            "Положи их в папку tessdata или установи русские/английские языковые данные Tesseract."
        )
    
    print(f"Tesseract executable: {TESSERACT_CMD}")
    print(f"Tessdata directory: {TESSDATA_DIR}")
    print(f"TESSDATA_PREFIX env: {os.environ.get('TESSDATA_PREFIX', 'not set')}")


def grab_primary_screen() -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def normalize_text(text: str) -> str:
    # Uppercase first to simplify matching.
    text = text.upper().replace("Ё", "Е")

    # Convert common Latin OCR lookalikes into Cyrillic.
    homoglyphs = str.maketrans({
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
    })
    text = text.translate(homoglyphs)

    # Keep only Cyrillic/Latin letters and digits.
    return re.sub(r"[^A-ZА-Я0-9]+", "", text)


def preprocess_variants(crop_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    scale = cfg.upscale
    resized = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    # Two variants: normal and inverted threshold.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(thresh)

    # Light morphology can help unify thin button text.
    kernel = np.ones((2, 2), np.uint8)
    thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    inv_open = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)

    return [
        ("thresh", thresh_open),
        ("inv", inv_open),
        ("gray", gray),
    ]


def is_target_word(text: str) -> bool:
    norm = normalize_text(text)
    return any(norm.startswith(prefix) for prefix in cfg.target_prefixes)


def ocr_pass(
    ocr_img: np.ndarray,
    psm: int,
    x0: int,
    y0: int = 0,
) -> List[Tuple[int, int, int, int, float, str]]:
    # NOTE: do NOT pass --tessdata-dir here — on Windows paths with Cyrillic/non-ASCII
    # characters break the CLI argument parsing.  TESSDATA_PREFIX env var is used instead.
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        ocr_img,
        lang=cfg.ocr_lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    boxes: List[Tuple[int, int, int, int, float, str]] = []
    if not data or not data.get("text"):
        return boxes

    n = len(data["text"])
    for i in range(n):
        text = str(data["text"][i]).strip()
        if not text:
            continue

        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0

        if conf < cfg.min_confidence:
            continue

        if not is_target_word(text):
            continue

        left = int(data["left"][i] / cfg.upscale) + x0
        top = int(data["top"][i] / cfg.upscale) + y0
        bw = int(data["width"][i] / cfg.upscale)
        bh = int(data["height"][i] / cfg.upscale)

        # Expand the OCR text box into a likely full button rectangle.
        button_w = max(int(bw * 3.0), 120)
        button_h = max(int(bh * 2.4), 30)

        cx = left + bw // 2
        cy = top + bh // 2
        x1 = max(x0, cx - button_w // 2)
        y1 = max(0, cy - button_h // 2)
        x2 = min(10**9, x1 + button_w)
        y2 = min(10**9, y1 + button_h)

        # Apply extra padding.
        x1 = max(x0, x1 - 26)
        y1 = max(0, y1 - 10)
        x2 = x2 + 26
        y2 = y2 + 10

        boxes.append((x1, y1, x2 - x1, y2 - y1, conf, text))

    return boxes


def dedupe_boxes(boxes: List[Tuple[int, int, int, int, float, str]]) -> List[Tuple[int, int, int, int, float, str]]:
    filtered: List[Tuple[int, int, int, int, float, str]] = []
    for box in boxes:
        x, y, bw, bh, conf, text = box
        keep = True
        for fx, fy, fw, fh, _, _ in filtered:
            ix1 = max(x, fx)
            iy1 = max(y, fy)
            ix2 = min(x + bw, fx + fw)
            iy2 = min(y + bh, fy + fh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = bw * bh
            farea = fw * fh
            if inter / float(min(area, farea) + 1e-6) > 0.4:
                keep = False
                break
        if keep:
            filtered.append(box)
    return filtered


# ── NEW: blue-color region detection ──────────────────────────────────────────

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


def ocr_on_region(
    frame: np.ndarray,
    rx: int,
    ry: int,
    rw: int,
    rh: int,
) -> List[Tuple[int, int, int, int, float, str]]:
    """Run OCR variants on a specific sub-region of the frame."""
    region_bgr = frame[ry: ry + rh, rx: rx + rw]
    if region_bgr.size == 0:
        return []

    variants = preprocess_variants(region_bgr)
    boxes: List[Tuple[int, int, int, int, float, str]] = []

    for variant_name, ocr_img in variants:
        for psm in cfg.psm_modes:
            try:
                found = ocr_pass(ocr_img, psm=psm, x0=rx, y0=ry)
                if found and cfg.print_ocr_debug:
                    print(f"    [color-ocr] variant={variant_name}, psm={psm} → {len(found)} hit(s)")
                boxes.extend(found)
            except Exception as e:
                print(f"    [color-ocr] failed variant={variant_name}, psm={psm}: {e}")

    return boxes


# ── main detection function ────────────────────────────────────────────────────

def find_buy_button_boxes(frame: np.ndarray) -> List[Tuple[int, int, int, int, float, str]]:
    h, w = frame.shape[:2]
    x0 = int(w * cfg.right_crop_ratio)
    crop = frame[:, x0:]

    all_boxes: List[Tuple[int, int, int, int, float, str]] = []

    # ── Pass 1: colour-guided OCR on detected blue regions ────────────────
    if cfg.use_color_detection:
        blue_regions = find_blue_regions(frame, x0)
        for rx, ry, rw, rh in blue_regions:
            found = ocr_on_region(frame, rx, ry, rw, rh)
            all_boxes.extend(found)

    # ── Pass 2: standard whole-crop OCR (original behaviour) ──────────────
    variants = preprocess_variants(crop)
    for variant_name, ocr_img in variants:
        for psm in cfg.psm_modes:
            try:
                boxes = ocr_pass(ocr_img, psm=psm, x0=x0)
                all_boxes.extend(boxes)
            except Exception as e:
                print(f"OCR pass failed for variant={variant_name}, psm={psm}: {e}")

    # ── Pass 3: generous full-screen fallback if still nothing ────────────
    if not all_boxes:
        print("  [fallback] no candidates yet – running full-screen OCR pass")
        full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, full_thresh = cv2.threshold(full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            all_boxes.extend(ocr_pass(full_thresh, psm=11, x0=0))
            all_boxes.extend(ocr_pass(cv2.bitwise_not(full_thresh), psm=11, x0=0))
        except Exception as e:
            print(f"Fallback OCR pass failed: {e}")

    if cfg.print_ocr_debug:
        print(f"Right crop starts at x={x0}, frame size={w}x{h}")
        total = len(all_boxes)
        print(f"  total raw candidates before dedup: {total}")
        for idx, (x, y, bw, bh, conf, text) in enumerate(all_boxes[:20], 1):
            print(f"  candidate {idx}: text={text!r}, conf={conf:.1f}, box=({x},{y},{bw},{bh})")

    return dedupe_boxes(all_boxes)


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
            "'Купить' not found",
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
    variants = preprocess_variants(crop)

    # Save the first preprocessing variant for visual debugging.
    debug = variants[0][1]
    cv2.imwrite(cfg.debug_output_path, debug)

    # Also save a blue-mask visualisation so it's easy to check colour tuning.
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
    try:
        ensure_tesseract_ready()
    except Exception as e:
        print(str(e))
        return 1

    print(f"Tesseract executable: {TESSERACT_CMD}")
    print(f"Tessdata directory: {TESSDATA_DIR}")
    print(f"Starting in {cfg.delay_seconds} seconds...")
    time.sleep(cfg.delay_seconds)

    frame = grab_primary_screen()
    if cfg.save_debug_image:
        try:
            save_debug_image(frame)
            print(f"Saved debug OCR image: {Path(cfg.debug_output_path).resolve()}")
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