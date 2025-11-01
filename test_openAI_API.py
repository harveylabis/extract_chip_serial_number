import base64
from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "images\\capture1\\WIN_20251027_18_30_00_Pro.jpg"

# you may include the range and nature (engraved, pen)

# Getting the Base64 string
base64_image = encode_image(image_path)


response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": "I am listing the serial number of IC package units ranging from 1-1000. What is the serial number of this unit? Return only the serial number because I will log it in Excel file." },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)

print(response.output_text)
