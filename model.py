# reference: https://www.hackersrealm.net/post/extract-features-from-image-python
# reference: https://github.com/yuukicammy/mit-adobe-fivek-dataset
from torch.utils.data.dataloader import DataLoader
from dataset.fivek import MITAboveFiveK
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Reshape
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import tensorflow
import matplotlib.pyplot as plt

def main():
    data_loader = DataLoader(
        MITAboveFiveK(root="path-to-dataset-root", split="debug", download=False, experts=["a"]),
        batch_size=None)

    VGG_model = VGG16()
    VGG_model = Model(inputs=VGG_model.inputs, outputs=VGG_model.layers[-2].output)

    CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    emotion_list = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]

    # {"amusement":[features_list], "awe":[features_list]}
    original_features = {}
    edited_features = {}
    # initialize the original_features and edited_features
    for emotion in emotion_list:
        original_features[emotion] = []
        edited_features[emotion] = []
    
    for item in data_loader:
        original_image_path = item["files"]["dng"]
        original_img = load_img(original_image_path, target_size=(224, 224))
        # classification using CLIP
        inputs_clip = CLIP_processor(text=["image evokes amusement", "image evokes awe", "image evokes contentment", "image evokes excitement", "image evokes anger", "image evokes disgust", "image evokes fear", "image evokes sadness"], images=original_img, return_tensors="pt", padding=True)
        outputs_clip = CLIP_model(**inputs_clip)
        logits_per_image = outputs_clip.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        # get the highest prob in probs list
        highest_prob = torch.max(probs)
        highest_prob_index = 0
        print(probs)
        for i in range(8):
            if (probs[0][i].item()==highest_prob):
                highest_prob_index = i
        image_class = emotion_list[highest_prob_index]
        
        original_img = img_to_array(original_img)
        original_img = original_img.reshape((1, original_img.shape[0], original_img.shape[1], original_img.shape[2]))
        original_img = preprocess_input(original_img)
        # extract original feature using pretrained VGG16 model
        original_feature = VGG_model.predict(original_img, verbose=0)
        original_features[image_class].append(original_feature)

        edited_image_path = item["files"]["tiff16"]["a"]
        edited_img = load_img(edited_image_path, target_size=(224, 224))
        edited_img = img_to_array(edited_img)
        edited_img = edited_img.reshape((1, edited_img.shape[0], edited_img.shape[1], edited_img.shape[2]))
        edited_img = preprocess_input(edited_img)
        # extract edtied feaure using pretrained VGG16 model
        edited_feature = VGG_model.predict(edited_img, verbose=0)
        edited_features[image_class].append(edited_feature)

    model_dict = {}
    for emotion in emotion_list:
        if(len(original_features[emotion])!=0):
            X = np.array(original_features[emotion])
            print(X)
            y = np.array(edited_features[emotion])
            print(y)

            # build model with fully connected layer
            model_dict[emotion] = Sequential([
                Dense(64, activation='relu', input_shape=(1, 4096)),
                Dense(32, activation='relu'),
                Dense(4096, activation='linear')
            ])

            print(model_dict[emotion].summary())
            model_dict[emotion].compile(optimizer='adam', loss='mean_squared_error')
            model_dict[emotion].fit(X, y, validation_split=0.2, epochs=10, batch_size=32)
        
    # get input from test dataset
    test_img = load_img("test.dng", target_size=(224, 224))
    # classify using CLIP
    logits_per_image = outputs_clip.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    # get the highest prob in probs list
    highest_prob = torch.max(probs)
    print(probs)
    highest_prob_index = 0
    for i in range(8):
        if (probs[0][i].item()==highest_prob):
            highest_prob_index = i
    image_class = emotion_list[highest_prob_index]
    if(image_class in model_dict.keys()):
        # get features
        test_img = img_to_array(test_img)
        test_img = test_img.reshape((1, test_img.shape[0], test_img.shape[1], test_img.shape[2]))
        test_img = preprocess_input(test_img)
        # extract original feature using pretrained VGG16 model
        test_feature = VGG_model.predict(test_img, verbose=0)
        test_X = tensorflow.reshape(np.array(test_feature), shape=(1,1,4096))
        predicted_feature = model_dict[image_class].predict(test_X)
        print(predicted_feature)
        predicted_feature = tensorflow.reshape(np.array(predicted_feature), shape=(1,1,4096))
        decoder = Sequential([
            Dense(256, activation='relu', input_shape=(1,4096)),
            Dense(512, activation='relu'),
            Dense(224*224, activation='sigmoid'),
            Reshape((224, 224))
        ])
        decode_image = decoder.predict(predicted_feature)
        plt.imshow(decode_image[0])
        plt.show()
    else:
        print("need larger dataset")


if __name__ == "__main__":
    main()