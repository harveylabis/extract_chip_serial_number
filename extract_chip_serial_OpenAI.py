import base64
from openai import OpenAI

client = OpenAI()

def encode_image_OpenAI(image_path):
    """Function to encode the image"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def get_response(prompt_to_input, )
    
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

                
# Path to your image
image_path = "images\\capture1\\WIN_20251027_18_31_49_Pro cropped.jpg"

# you may include the range and nature (engraved, pen)
range_input = "1-3000"
marking_nature = "engraved"
generated_prompt = generate_prompt(range=range_input, nature=marking_nature)

base64_image = encode_image_OpenAI(image_path)


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

print("Prompt: ", generated_prompt)
print("Image input: ", image_path)
print("Serial number found: ", response.output_text)

## TESTING SCRIPT

# print("prompt1: ", generate_prompt(range="1-1000", nature="engraved"))
# print("prompt2: ", generate_prompt(range="1-1000"))
# print("prompt3: ", generate_prompt(nature="engraved"))
# print("prompt4: ", generate_prompt())

