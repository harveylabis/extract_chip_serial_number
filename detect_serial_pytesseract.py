import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt  # only needed if using Jupyter/Colab

# Configure this if needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def show_image(title, img):
    """Display image (works both in notebook or local)."""
    cv2.imshow(title, img)
    cv2.waitKey(0)

def get_chip_rois(img, rows=2, cols=3):
    """Splits the image into a fixed grid of chip ROIs."""
    h, w = img.shape[:2]
    h_step, w_step = h // rows, w // cols
    rois = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * h_step, (r + 1) * h_step
            x1, x2 = c * w_step, (c + 1) * w_step
            rois.append(img[y1:y2, x1:x2])
    return rois

def crop_bottom_right(chip_img, frac_h=0.4, frac_w=0.4):
    h, w = chip_img.shape[:2]
    return chip_img[int((1-frac_h)*h):h, int((1-frac_w)*w):w]

def preprocess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Simple clean threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Reduce speckles
    thresh = cv2.medianBlur(thresh, 3)
    return thresh

def ocr_number(roi):
    # Upscale for better OCR accuracy
    roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    config = '--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(roi, config=config)
    return ''.join(filter(str.isdigit, text))



# Load image
img = cv2.imread("images\chatgpt_image.png")

# 1. Extract 6 chips
chips = get_chip_rois(img)

serials = []
for idx, chip in enumerate(chips):
    print(f"\n==== Chip {idx+1} ====")

    # Show full chip
    cv2.imshow(f"Chip {idx+1}", chip)
    cv2.waitKey(0)

    # 2. Crop bottom-right section
    roi = crop_bottom_right(chip)

    # Show ROI
    cv2.imshow(f"Chip {idx+1} - Bottom Right ROI", roi)
    cv2.waitKey(0)

    # 3. Preprocess for OCR
    processed = preprocess(roi)

    # Show processed image
    cv2.imshow(f"Chip {idx+1} - Processed (Binary)", processed)
    cv2.waitKey(0)

    # 4. OCR
    number = ocr_number(processed)
    print(f"Detected number: {number}")

    serials.append(number)

cv2.destroyAllWindows()

# Display results in 2x3 format
print("\nFinal Detected Serial Numbers:")
print(serials[0:3])
print(serials[3:6])