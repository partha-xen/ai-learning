import base64
from openai import OpenAI

client = OpenAI()

prompt = "What is in this image?"
with open("goog.png", "rb") as image_file:
    b64_image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64_image}",
                },
            ],
        }
    ],
    temperature=0,
)

print(response.output_text)
