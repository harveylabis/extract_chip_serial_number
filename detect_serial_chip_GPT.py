import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Configure this if needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def image_to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def crop_image(image, method='contour'):
    if method == 'contour':
        """This needs to be updated"""
        gray = cv.equalizeHist(image)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thresh = 255 - thresh  # make chip white
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        chip_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(chip_contour)
        cropped_image = img[y:y+h, x:x+w]
    else:
        h, w = image.shape[:2]
        x0, x1 = int(w*0.1), int(w*0.72)
        y0, y1 = int(h*0.15), int(h*0.8)
        cropped_image = img[y0:y1, x0:x1]

    return cropped_image

def apply_clahe(image):
    """
    clipLimit=3.0: controls how much contrast can be increased
        (higher → stronger enhancement, but may introduce noise)
    tileGridSize=(8,8): divides the image into local grids of 8 by 8 pixels each"""
    
    # for clip in [1.0, 2.0, 3.0, 4.0, 5.0]:
    #     temp_clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    #     temp_result = temp_clahe.apply(image)
    #     plt.imshow(temp_result, cmap='gray')
    #     plt.title(f"clipLimit={clip}")
    #     plt.show()

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(image)

def apply_GaussianBlur(image):
    """Gaussian blur = convolution with a Gaussian kernel. 
    It’s a low-pass filter: attenuates high-frequency components 
    (noise, speckle) while preserving lower-frequency content 
    (large shapes, text strokes if kernel small)."""

    blurred= cv.GaussianBlur(image, (3, 3), 0)

    # kernels = [(3,3), (5,5), (7,7)]
    # for k in kernels:
    #     b = cv.GaussianBlur(image, k, 0)
    #     cv.imshow(f"blur_{k}", b)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # # we can also try Bilateral alternative (preserve edges)
    # blurred = cv.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)

    # # we can also try Median for salt-and-pepper
    # blurred = med = cv.medianBlur(image, 3)  # kernel must be odd

    return blurred

def apply_bilateralFilterBlur(image):
    return cv.bilateralFilter(image, d=7, sigmaColor=75, sigmaSpace=75)

def apply_unsharp(image):

    """Unsharp masking is a sharpening technique that adds back high-frequency detail (edges) by 
    subtracting a blurred version of the image from the original, then adding that difference back into the original.
    We use it because after smoothing (Gaussian blur) and CLAHE, text edges may be softened or 
    low-contrast — unsharp brings stroke edges back while leaving low-frequency background nearly unchanged. 
    That increases OCR legibility."""

    sigma = 1.0
    amount = 1.4
    ksize = (0,0)  # let sigma define kernel size

    blur = cv.GaussianBlur(image, ksize, sigmaX=sigma) # this is adding blur to img_clahr
    mask = cv.subtract(image, blur)

    # optional: ignore tiny mask values (suppress noise)
    thresh_val = 8
    _, mask_thresh = cv.threshold(mask, thresh_val, 255, cv.THRESH_TOZERO)

    # scale mask and add back
    mask_scaled = (mask_thresh.astype(np.float32) * amount).clip(0,255).astype(np.uint8)
    sharpened = cv.add(image, mask_scaled)

    return sharpened

def apply_tophat(image):
    """Top-hat highlights bright details (like your laser-etched letters)
    while suppressing slow background intensity variations (like uneven chip surface illumination). """
    
    # sizes = [15, 35, 55]
    # for k in sizes:
    #     kernel = cv.getStructuringElement(cv.MORPH_RECT, (k,k))
    #     tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
    #     plt.imshow(tophat, cmap='gray')
    #     plt.title(f'Top-hat kernel={k}x{k}')
    #     plt.axis('off')
    #     plt.show()

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (35, 35))
    tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)

    return tophat    

def apply_OtsuInvThreshold(image):
    """Separate foreground (chip text) from background (chip surface).
    Otsu looks at the image histogram and finds the “best split point” between dark and bright pixels."""
    
    # For OCR and contour detection, we often want text white (foreground) and background black. Hence, BINARY_INV
    _, otsu_inv = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    return otsu_inv 

def apply_adaptiveThreshold(image):
    """Instead of one global threshold, Adaptive thresholding computes a local threshold for each small region (neighborhood) of the image.
    In Adaptive Gaussian, the threshold for a pixel = weighted mean of its neighborhood (Gaussian weights) − a constant C."""
    adapt_inv = cv.adaptiveThreshold(
    image,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    31,   # block size (neighborhood window)
    10    # constant C subtracted from mean
    )

    return adapt_inv

def apply_morphing(image):
    """Operation	    Order	Purpose	                        Typical Result
       Closing	         1st	Fill gaps, connect strokes	    Smooth continuous text
       Opening	         2nd	Remove speckles/noise	        Clean background"""

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    # Fill small gaps
    closed = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=1)
    # Remove small noise
    opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel, iterations=1)

    return opened


def apply_laplacian_edges(image):
    """The Laplacian is a second-order derivative operator used to find edges — areas of rapid intensity change.
    While Sobel and Canny focus on the directional gradients (edges along X and Y axes), 
    the Laplacian detects overall intensity curvature — both horizontal and vertical — at once."""
    # Apply Laplacian edge detection
    lap = cv.Laplacian(image, cv.CV_16S, ksize=3)
    lap = cv.convertScaleAbs(lap)
    combined = cv.addWeighted(image, 0.8, lap, 0.2, 0)

    return combined

def combine_laplacian_threshold(img_lap, img_thres):
    """Combine your Laplacian edge image (fine text outlines) with your 
    clean thresholded binary image (solid text regions),"""

    # Ensure same dimensions (just in case)
    h, w = img_thres.shape
    lap = cv.resize(img_lap, (w, h))

    # Combine with weighted blending
    combined = cv.addWeighted(img_thres, 0.8, lap, 0.2, 0)

    # Optional denoise slightly to reduce halos
    combined = cv.medianBlur(combined, 3)

    # # Display result
    # plt.imshow(combined, cmap='gray')
    # plt.title("Combined for OCR (Threshold + Laplacian)")
    # plt.axis('off')
    # plt.show()

    return combined

def read_ocr(image):
    # Resize for better OCR accuracy (optional)
    scale = 2.0
    combined_scaled = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

    # Configure Tesseract
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Perform OCR
    text = pytesseract.image_to_string(combined_scaled, config=custom_config)

    print("OCR Output:")
    print(text.strip())



# load the image
img = cv.imread('images\\capture1\\WIN_20251027_18_37_09_Pro serial.jpg')

# crop - can be ratio-based or contour-based
# img_roi_ratio = crop_image(img, 'ratio')

# convert to grayscale
# gray = image_to_grayscale(img_roi_ratio)
gray = image_to_grayscale(img)

# apply CLAHE
img_clahe = apply_clahe(gray)

# apply gaussian blur - blur smooths everything equally
img_gaus_blurred = apply_GaussianBlur(img_clahe)

# apply bilateral filter blur - Bilateral filtering is edge-preserving
img_bilat_blurred = apply_bilateralFilterBlur(img_clahe)

# unsharp
img_unsharped = apply_unsharp(img_clahe)

# tophat
img_tophat = apply_tophat(img_unsharped)

# Otsu thresholding 
img_OtsuThreshold = apply_OtsuInvThreshold(img_tophat)

# Adaptive Gaussian thresholding
img_AdaptiveThreshold = apply_adaptiveThreshold(img_tophat)

# apply morphing (close then open)
img_morphing = apply_morphing(img_AdaptiveThreshold)

# apply laplacian edges
img_laplacian_edges = apply_laplacian_edges(img_tophat)

# combine laplacian and threshold
img_combined_laplacian_threshold = combine_laplacian_threshold(img_laplacian_edges, img_AdaptiveThreshold)


plt.imshow(img_OtsuThreshold, cmap='gray')
plt.show()








