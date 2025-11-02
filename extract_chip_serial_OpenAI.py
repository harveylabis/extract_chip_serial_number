import base64
from openai import OpenAI
import cv2 as cv
import matplotlib.pyplot as plt

client = OpenAI()

def encode_image_OpenAI(image_path):
    """Function to encode the image"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def get_serial_number_OpenAI(generated_prompt, base64_image):
    """Ask OpenAI for the serial number using the prompt and image provided."""
    response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": generated_prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
                ],
            }
        ],
    )

    return response.output_text
    
def generate_prompt(range="", nature=""):
    """Generate the prompt to input."""
    goal_template = "I am listing the serial number of IC package units"
    range_template = f", ranging from {range}"
    nature_template = f" The serial number are marked on the units by {nature}. "
    question_template = "What is the serial number of this unit? Return only the serial number because I will log it in Excel file."

    if range and nature:
        prompt = f"{goal_template}{range_template}." + nature_template + question_template
    elif range and nature=='':
        prompt = f"{goal_template}{range_template}. " + question_template
    elif range=='' and nature:
        prompt = f"{goal_template}." + f"{nature_template} " + question_template
    else:
        prompt = goal_template + '. ' + question_template

    return prompt  

def get_ROI_from_ROIs(image_path, quadrant=4):
    """Get the ROIs from a single image.
    Avoid sending the whole image, only the ROI."""
    rois = get_chip_quadrants(image_path)
    if quadrant==1:
        roi = rois[1]
    elif quadrant==2:
        roi = rois[0]
    elif quadrant==3:
        roi = rois[2]
    elif quadrant==4:
        roi = rois[3]

    return roi

def crop_image(image, frac_h=0.9, frac_w=0.9):
    """Crop the image if necessary"""
    h, w = image.shape[:2]
    return image[int((1-frac_h)*h):h, int((1-frac_w)*w):w]

def get_chip_quadrants(image, rows=2, cols=2):
    """Splits the image into a fixed grid of chip ROIs. Default is 2x2."""
    h, w = image.shape[:2]
    h_step, w_step = h // rows, w // cols
    rois = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * h_step, (r + 1) * h_step
            x1, x2 = c * w_step, (c + 1) * w_step
            rois.append(image[y1:y2, x1:x2])
    
    return rois

def apply_clahe_bilateralFilter(image):
    """
    clipLimit=3.0: controls how much contrast can be increased
        (higher → stronger enhancement, but may introduce noise)
    tileGridSize=(8,8): divides the image into local grids of 8 by 8 pixels each"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
    img_clahe = clahe.apply(gray)

    # Smooth out noise
    img_smooth = cv.bilateralFilter(img_clahe, 5, 30, 30)

    # Merge back to 3-channel for upload
    img_final = cv.cvtColor(img_smooth, cv.COLOR_GRAY2BGR)

    return img_final


###################  PROGRAM LOGIC STARTS HERE ############################
                
# Path to your image
image_path = "image1.jpg"

# get image (or capture image)
image_read = cv.imread(image_path)  # Reads as BGR by default
image_rgb = cv.cvtColor(image_read, cv.COLOR_BGR2RGB) # Convert to RGB, plt expects RGB

# cropped image if necessary = get ROI
image_ROI = get_ROI_from_ROIs(image_rgb, quadrant=4) # only if serial is in quadrant X; OpenCV default is RGB
image_ROI_BGR = cv.cvtColor(image_ROI, cv.COLOR_RGB2BGR) # we want ro show BGR for engraved markings to stand out as blue

# # apply CLAHE (optional depending on the image)
# image_upload_clahe = apply_clahe_bilateralFilter(image_ROI)
# plt.subplot(133),plt.imshow(image_upload_clahe)
# plt.title('CLAHE')

# save the image to be sent to OpenAI
image_to_send_OpenAI = f"{image_path} cropped.png"
cv.imwrite(image_to_send_OpenAI, image_ROI)

# you may include the range and nature (engraved, pen)
range_input = "1-3000"
marking_nature = "engraved"
generated_prompt = generate_prompt(range=range_input, nature=marking_nature)

# prepare to upload the image to OpenAI 
base64_image = encode_image_OpenAI(image_to_send_OpenAI)

# send the image and prompt OpenAI, then get the response
serial_number = get_serial_number_OpenAI(generated_prompt, base64_image)

# printing the output to screen
print("Prompt: ", generated_prompt)
print("Image input: ", image_to_send_OpenAI)
print("Serial number found: ", serial_number)

# plotting the images
plt.subplot(131),plt.imshow(image_rgb)
plt.title('Captured image')
plt.subplot(132),plt.imshow(image_ROI_BGR)
plt.title('Region of Interest (ROI) in BGR Color Map')
plt.subplot(133)
plt.axis('off')  # hide axes
plt.text(0.5, 0.5, serial_number,
         fontsize=50,
         color='yellow',
         fontweight='bold',
         ha='center', va='center',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='black', alpha=0.4, pad=15, edgecolor='none'))
plt.title('Detected Serial Number', y=0.7, pad=0)
plt.tight_layout()

plt.show()


