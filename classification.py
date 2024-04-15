import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image1 = Image.open("MIT-Adobe_5K_dataset/jpeg/a0019.jpeg")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image19: ",probs1)

image1 = Image.open("MIT-Adobe_5K_dataset/jpeg/a0026.jpeg")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image26: ",probs1)

image1 = Image.open("MIT-Adobe_5K_dataset/jpeg/a0031.jpeg")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image31: ",probs1)

image1 = Image.open("MIT-Adobe_5K_dataset/jpeg/a0033.jpeg")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image33: ",probs1)

image1 = Image.open("MIT-Adobe_5K_dataset/jpeg/a0035.jpeg")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image35: ",probs1)