import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('images\\test_image.jpg', cv.IMREAD_GRAYSCALE) # image to check 
assert img is not None, "file could not be read, check with os.path.exists()"
print(f"image shape: {img.shape}")

template_paths = ['template_matching\\templates\\5.png',
                  'template_matching\\templates\\6.png',
                  'template_matching\\templates\\7.png',
                  'template_matching\\templates\\8.png']

for template in template_paths:
    # temp_img = cv.resize(cv.imread(template, cv.IMREAD_GRAYSCALE), (50, 30))
    temp_img =(cv.imread(template, cv.IMREAD_GRAYSCALE))
    print(f"template {template} shap# e: {temp_img.shape}")
    print(f"template {template} shape: {temp_img.shape}")
    # plt.imshow(temp_img)
    # plt.title(f"Template: {os.path.basename(template)}")
    # plt.show()

    # get h, w of template
    w, h = temp_img.shape[::-1]
    img2 = img.copy()
    method = getattr(cv, 'TM_CCORR_NORMED')
    result = cv.matchTemplate(img2, temp_img, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    
    print(f"{template}:: min_loc={min_loc}, max_loc={max_loc}")
    print(f"{template}:: min_val={min_val}, max_val={max_val}")

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    threshold = 0.94 # very special number
    if max_val > threshold:
        cv.rectangle(img,top_left, bottom_right, 255, 2)
        plt.subplot(121),plt.imshow(result,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(method)
        print(f"Template {template} FOUND in image with sufficient confidence (max_val={max_val})")

    else:
        print(f"Template {template} not found in image with sufficient confidence (max_val={max_val})")
        print()

plt.show()


