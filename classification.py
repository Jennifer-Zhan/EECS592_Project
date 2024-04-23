import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image1 = Image.open("image1.png")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image1: ",probs1)

image1 = Image.open("image2.png")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image2: ",probs1)

image1 = Image.open("image3.png")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image3: ",probs1)

image1 = Image.open("image4.png")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image4: ",probs1)

image1 = Image.open("image5.png")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image5: ",probs1)

image1 = Image.open("image6.png")
inputs1 = processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=image1, return_tensors="pt", padding=True)
outputs1 = model(**inputs1)
#print(outputs)
logits_per_image1 = outputs1.logits_per_image
probs1 = logits_per_image1.softmax(dim=1)
print("probs for image6: ",probs1)