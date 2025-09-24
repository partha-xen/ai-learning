from transformers import AutoModel, AutoTokenizer
from pprint import pprint


# classifier = pipeline("zero-shot-classification")
# result = classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"],
# )

# print(result)

# generator = pipeline("text-generation")
# result = generator("In this course, we will teach you how to")
# print(result)

# ner = pipeline("ner", model="dslim/bert-base-NER")
# result = ner("My name is Wolfgang and I live in Berlin")
# print(result)

# translator = pipeline("translation_en_to_de")
# result = translator("I love programming")
# print(result)

# unmasker = pipeline("fill-mask", model="bert-base-uncased")
# result = unmasker("This man works as a [MASK].")
# print([r["token_str"] for r in result])

# result = unmasker("This woman works as a [MASK].")
# print([r["token_str"] for r in result])

# # List available pipeline types
# print("Available pipeline types:")
# for task in PIPELINE_REGISTRY.get_supported_tasks():
#     print(f"  - {task}")

# # Alternative way to see pipeline registry contents
# print(f"\nTotal pipeline types: {len(PIPELINE_REGISTRY.get_supported_tasks())}")

# # Print some common default checkpoints (without downloading)
# print("\nCommon default checkpoints for popular pipelines:")
# default_models = {
#     "sentiment-analysis": "distilbert-base-uncased-finetuned-sst-2-english",
#     "text-generation": "gpt2",
#     "fill-mask": "bert-base-uncased",
#     "ner": "dbmdz/bert-large-cased-finetuned-conll03-english",
#     "question-answering": "distilbert-base-cased-distilled-squad",
#     "summarization": "sshleifer/distilbart-cnn-12-6",
#     "translation": "t5-base",
#     "text-classification": "distilbert-base-uncased",
#     "zero-shot-classification": "facebook/bart-large-mnli",
#     "automatic-speech-recognition": "facebook/wav2vec2-base-960h",
#     "image-classification": "google/vit-base-patch16-224",
#     "object-detection": "facebook/detr-resnet-50",
#     "image-segmentation": "facebook/detr-resnet-50-panoptic",
# }

# for task, model in default_models.items():
#     if task in PIPELINE_REGISTRY.get_supported_tasks():
#         print(f"  {task}: {model}")

# print(f"\nNote: These are common default models. Actual defaults may vary by version.")

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
pprint(inputs)

tokens = tokenizer.tokenize(raw_inputs)
print("Tokens:")
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print("IDs:")
print(ids)

outputs = model(**inputs)
# print("Outputs:")
# print(outputs["last_hidden_state"])
# print("Last hidden state shape:")
# print(outputs.last_hidden_state.shape)
# print("Last hidden state:")
# print(outputs.last_hidden_state)

print(tokenizer.decode(inputs["input_ids"][0]))
print(tokenizer.decode(inputs["input_ids"][1]))
