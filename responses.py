import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-4o",
    instructions="You are a supply chain expert that helps with logistics and transportation.",
    input="How do I check if a shipment is delayed?",
)

print(response.output_text)
