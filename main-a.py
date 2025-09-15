from openai import OpenAI
import json
import os

client = OpenAI()

prompt = "Return only JSON with the Python function source."

schema = {
    "name": "code_payload",
    "schema": {
        "type": "object",
        "properties": {"filename": {"type": "string"}, "source": {"type": "string"}},
        "required": ["filename", "source"],
        "additionalProperties": False,
    },
    "strict": True,
}

resp = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": (
                "Produce a single Python function 'bmi(weight_kg: float, height_cm: float) -> float' "
                "with docstring and type hints. Round to 2 decimals. "
                "Respond ONLY as JSON: {'filename': 'bmi.py', 'source': '<code>'}."
            ),
        }
    ],
    # output_format={"type": "json_schema", "json_schema": schema},  # Structured Outputs
    temperature=0,
    max_output_tokens=800,
)

payload = resp.output[0].content[0].text  # JSON string
print("Payload", payload)
print()
# clean up ```json ... ```
if payload.startswith("```"):
    payload = payload.strip("`")
    payload = payload.replace("json\n", "", 1).replace("\n```", "")

# load it
data = json.loads(payload)
os.makedirs("generated", exist_ok=True)
with open(f"generated/{data['filename']}", "w") as f:
    f.write(data["source"])
print("Wrote", f"generated/{data['filename']}")
