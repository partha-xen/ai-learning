import base64
from pathlib import Path


def file_to_base64(path: str) -> str:
    file_bytes = Path(path).read_bytes()  # read file as bytes
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return encoded


# Example: encode an image
b64_str = file_to_base64("goog.png")
print(b64_str[:100], "...")  # show only first 100 chars


def base64_to_file(b64_str: str, output_path: str):
    file_bytes = base64.b64decode(b64_str.encode("utf-8"))
    Path(output_path).write_bytes(file_bytes)


# Example: restore the image
base64_to_file(b64_str, "restored.png")
