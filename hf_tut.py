from transformers import pipeline

# classifier = pipeline("zero-shot-classification")
# result = classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"],
# )

# print(result)

# generator = pipeline("text-generation")
# result = generator("In this course, we will teach you how to")
# print(result)

ner = pipeline("ner", model="dslim/bert-base-NER")
result = ner("My name is Wolfgang and I live in Berlin")
print(result)

translator = pipeline("translation_en_to_de")
result = translator("I love programming")
print(result)
