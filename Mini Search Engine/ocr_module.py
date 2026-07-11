import cv2
import pytesseract
import numpy as np
import os

# Auto-configure tesseract path for Windows users if it's in the default installation directory
tess_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tess_path):
    pytesseract.pytesseract.tesseract_cmd = tess_path


def _run_tesseract(image, psm):
    """Run Tesseract with the given PSM mode and return cleaned text."""
    config = (
        f'--oem 3 --psm {psm} '
        '-c tessedit_char_whitelist='
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    )
    raw = pytesseract.image_to_string(image, config=config)
    return "".join(filter(str.isalnum, raw)).lower().strip()


def extract_text_from_canvas(canvas):
    """
    Extract text from the air-writing canvas.
    The canvas uses neon cyan strokes on a black background.
    We convert to clean white-on-black, upscale, and run Tesseract with
    multiple PSM modes to maximise accuracy for both single characters
    and whole words.
    """
    try:
        # Convert neon-coloured canvas to grayscale
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Find the bounding box of the drawn regions
        coords = cv2.findNonZero(gray)
        if coords is None:
            return "", None  # Nothing was drawn

        x, y, w, h = cv2.boundingRect(coords)

        # Crop the drawing out, adding generous padding
        pad = 40
        roi = gray[max(0, y - pad):min(canvas.shape[0], y + h + pad),
                    max(0, x - pad):min(canvas.shape[1], x + w + pad)]

        # Upscale 4x — significantly improves Tesseract accuracy on handwriting
        roi = cv2.resize(roi, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)

        # Gaussian blur to smooth jagged air-written edges
        roi = cv2.GaussianBlur(roi, (5, 5), 1.5)

        # Threshold to binary white-on-black
        _, thresh = cv2.threshold(roi, 12, 255, cv2.THRESH_BINARY)

        # Morphological operations: thicken strokes and close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thickened = cv2.dilate(thresh, kernel, iterations=3)
        cleaned = cv2.morphologyEx(thickened, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Invert to black text on white background (required by Tesseract)
        cleaned = cv2.bitwise_not(cleaned)

        # Add generous white border padding (Tesseract needs whitespace around text)
        cleaned = cv2.copyMakeBorder(cleaned, 30, 30, 30, 30,
                                     cv2.BORDER_CONSTANT, value=255)

        # Try multiple PSM modes and pick the best via majority vote:
        # PSM 10 = single character (best for 1 letter)
        # PSM 8  = single word  (good for short words)
        # PSM 7  = single text line (good for longer words)
        results = {}  # text → count
        psm_order = [10, 8, 7]
        first_result = ""
        for psm in psm_order:
            txt = _run_tesseract(cleaned, psm)
            if txt:
                if not first_result:
                    first_result = txt  # PSM 10 result (highest priority)
                results[txt] = results.get(txt, 0) + 1

        if results:
            # Majority vote: pick the text that most modes agreed on
            best = max(results, key=lambda k: results[k])
            # If there's a tie (all different), prefer PSM 10's result
            # since air writing typically produces single chars/short words
            if results[best] == 1 and first_result:
                best = first_result
            print(f"OCR results: {results} → best: '{best}'")
            return best, None
        else:
            return "", None

    except pytesseract.TesseractNotFoundError:
        return None, "Error: Tesseract OCR is not installed or not in PATH. Please install it."
    except Exception as e:
        return None, f"OCR Error: {str(e)}"
