import cv2

img = cv2.imread("images\\capture1\\WIN_20251027_18_37_09_Pro serial filtered2.png")

# Try several scales
for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"downscale_{int(scale*100)}.png", resized)